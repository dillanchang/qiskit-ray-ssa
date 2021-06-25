# Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import logging
import numpy as np
import pandas as pd
from tletools import TLE
from astropy import units as u
import spacetrack.operators as op
from spacetrack import SpaceTrackClient


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class SpaceTrackETL:
    """A class for ETLing data from USSTRATCOM's space-track.org API

    :param stc: The client to use to make requests to the space-track.org API
    :type stc: spacetrack.SpaceTrackClient
    """
    def __init__(self, stc):
        self.stc = stc

    def get_leo_aso_catalog(self, norad_ids=None):
        """Retrieves entries from the Space Track Satellite Catalog for all ASOs
        that are in low Earth orbit.

        :param norad_ids: An optional list of NORAD IDs to fetch the TLEs
            for.  If NORAD IDs are not provided then data will be fetched
            for all ASOs in LEO.
        :type norad_ids: [str]

        :return: The catalog entries for LEO ASOs
        :rtype: pandas.DataFrame
        """
        query_params = {
            'decay': None,
            'current': 'Y'
        }
        if norad_ids:
            query_params['norad_cat_id'] = norad_ids
        else:
            query_params['period'] = op.less_than(128),
        leo_asos = self.stc.satcat(**query_params)
        return pd.DataFrame(leo_asos)

    def get_object_types(self, aso_df):
        """Normalizes the object type and NORAD ID columns from the Space Track
        Satellite Catalog DataFrame.  This will be used to add the object type
        to the TLE data.

        :param aso_df: The DataFrame composed of entries from the Space Track
            Satellite Catalog
        :type aso_df: pandas.DataFrame

        :return: A pandas DataFrame containing the ASO's ID and normalized
            object type
        :rtype: pandas.DataFrame
        """
        cols = ['NORAD_CAT_ID', 'OBJECT_TYPE']
        col_mapper = {'NORAD_CAT_ID': 'aso_id', 'OBJECT_TYPE': 'object_type'}
        # Standardize the column names
        object_types = aso_df[cols].rename(columns=col_mapper)

        # Lowercase the object type strings and replace spaces with underscores
        def norm_str(s):
            return s.lower().replace(' ', '_')

        object_types['object_type'] = object_types.object_type.apply(norm_str)
        return object_types

    def get_leo_tles_str(self, norad_ids, last_n_days, only_latest):
        """Uses the Space Track TLE API to get all the TLEs for the ASOs specified
        by the `norad_ids` for the specified time period.

        :param norad_ids: The NORAD IDs of the ASOs to get the TLEs for
        :type norad_ids: [str]

        :param last_n_days: The number of past days to get TLEs for
        :type last_n_days: int

        :param only_latest: Whether or not to only fetch the latest TLE for
            each ASO
        :type only_latest: bool

        :return: The three line elements for each specified ASO
        :rtype: str
        """
        query_params = {
            'epoch': f'>now-{last_n_days}',
            'norad_cat_id': norad_ids,
            'format': '3le'
        }
        if only_latest:
            query_params['ordinal'] = 1
            leo_tles_str = self.stc.tle_latest(**query_params)
        else:
            leo_tles_str = self.stc.tle(**query_params)

        return leo_tles_str

    def get_tles(self, raw_tle_str):
        """Parses the raw TLE string and converts it to TLE objects.

        :param raw_tle_str: The raw string form of the TLEs
        :type raw_tle_str: str

        :return: The parsed object representations of the TLEs
        :rtype: [tletools.TLE]
        """
        all_tle_lines = raw_tle_str.strip().splitlines()
        tles = []
        for i in range(len(all_tle_lines)//3):
            # Calculate offset
            j = i*3
            tle_lines = all_tle_lines[j:j+3]
            # Strip line number from object name line
            tle_lines[0] = tle_lines[0][2:]
            tle = TLE.from_lines(*tle_lines)
            tles.append(tle)
        return tles

    def get_aso_data(self, tles):
        """Extracts the necessary data from the TLE objects for doing orbital
        prediction.

        :param tles: The list of TLE objects to extract orbit information from
        :type tles: [tletools.TLE]

        :return: A DataFrame of the extracted TLE data
        :rtype: pandas.DataFrame
        """
        tles_data = []
        for tle in tles:
            aso_data = {}
            aso_data['aso_name'] = tle.name
            aso_data['aso_id'] = tle.norad
            aso_data['epoch'] = tle.epoch.to_datetime()
            # Convert the TLE object to a poliastro.twobody.Orbit instance
            orbit = tle.to_orbit()
            # Calculate the position and velocity vectors
            r, v = orbit.rv()
            # Convert position vector from kilometers to meters
            r_m = r.to(u.m).to_value()
            # Convert the velocity vector from km/s to m/s
            v_ms = v.to(u.m/u.s).to_value()
            # Extract the components of the state vectiors
            aso_data['r_x'], aso_data['r_y'], aso_data['r_z'] = r_m
            aso_data['v_x'], aso_data['v_y'], aso_data['v_z'] = v_ms
            tles_data.append(aso_data)
        return pd.DataFrame(tles_data)

    def build_leo_df(self, norad_ids, last_n_days, only_latest):
        """Builds a pandas DataFrame of LEO ASO orbit observations from data
        provided by USSTRATCOM via space-track.org.

        :param norad_ids: An optional list of NORAD IDs to fetch the TLEs
            for.  If NORAD IDs are not provided then data will be fetched
            for all ASOs in LEO.
        :type norad_ids: [str]

        :param last_n_days: The number of past days to get TLEs for
        :type last_n_days: int

        :param only_latest: Whether or not to only fetch the latest TLE for
            each ASO
        :type only_latest: bool

        :return: The Space Track orbit data for LEO ASOs
        :rtype: pandas.DataFrame

        """
        logger.info('Fetching Satellite Catalog Data...')
        leo_asos = self.get_leo_aso_catalog(norad_ids)
        norad_ids = leo_asos['NORAD_CAT_ID']
        # The space-track.org API is rate limited and the response size
        # of the data is capped.  Experimenting found that we can reliably
        # get successful responses for about 500 ASOs so we break the
        # NORAD IDs into chunks for processing.
        n_chunks = len(norad_ids) // 500
        if n_chunks > 1:
            norad_chunks = np.array_split(norad_ids, n_chunks)
        else:
            norad_chunks = [norad_ids]

        logger.info(f'Number of TLE Batch Requests: {len(norad_chunks)}')

        leo_tles = []
        logger.info('Starting to fetch TLEs from space-track.org')
        for idx, norad_chunk in enumerate(norad_chunks):
            logger.info(f'Processing batch {idx+1}/{len(norad_chunks)}')
            logger.info(f'Fetching TLEs for {len(norad_chunk)} ASOs...')
            aso_ids = norad_chunk.to_list()
            chunk_tle_str = self.get_leo_tles_str(aso_ids,
                                                  last_n_days,
                                                  only_latest)
            logger.info('Parsing raw TLE data...')
            chunk_tles = self.get_tles(chunk_tle_str)
            leo_tles += chunk_tles
        logger.info('Finished fetching TLEs')
        tle_cnt = len(leo_tles)
        logger.info(f'Calculating orbital state vectors for {tle_cnt} TLEs...')
        aso_data = self.get_aso_data(leo_tles)
        object_types = self.get_object_types(leo_asos)
        aso_data = aso_data.merge(object_types, on='aso_id', how='left')
        return aso_data


def st_callback(until):
    """Log the number of seconds the program is sleeping to stay
    within Space Track's API rate limit.

    :param until: The time the program will sleep til
    :type until: float
    """
    duration = int(round(until - time.time()))
    logger.info(f'Sleeping for {duration} seconds.')


def build_space_track_client(username, password, log_delay=True):
    """Builds a client for the space-track.org API.

    :param username: The space-track.org username
    :type username: str

    :param password: The space-track.org password
    :type password: str

    :param log_delay: Whether or not to log when the API client sleeps to
        prevent rate limiting
    :type log_delay: bool
    """
    stc = SpaceTrackClient(identity=username,
                           password=password)
    if log_delay:
        # Add a call back to the space track client that logs
        # when the client has to sleep to abide by the rate limits
        stc.callback = st_callback
    return stc


def run(args):
    """Runs the ETL process against the space-track.org API using the arguments
    supplied by the CLI.

    :param args: The command line arguments
    :type args: argparse.Namespace
    """
    space_track_client = build_space_track_client(args.st_user,
                                                  args.st_password)
    st_etl = SpaceTrackETL(space_track_client)

    orbit_data_df = st_etl.build_leo_df(norad_ids=args.norad_ids,
                                        last_n_days=args.last_n_days,
                                        only_latest=args.only_latest)
    logger.info('Serializing data...')
    orbit_data_df.to_parquet(args.output_path)

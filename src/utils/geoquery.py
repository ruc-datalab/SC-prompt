from typing import Optional, List, Dict, Callable
import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .dataset import normalize, combine_SC
from .args import *
from .trainer import Seq2SeqTrainer, EvalPrediction
from .bridge_content_encoder import get_database_matches, get_column_picklist
from .get_tables import dump_db_json_schema
import random
from .process_sql import get_schema, Schema, get_sql
from .evaluation import build_foreign_key_map_from_json, evaluate
import os
import json

sql_clauses = ['select', 'from', 'where', 'group', 'having', 'order', 'limit', 'intersect', 'union', 'except']
sql_ops_space = ['>', '<', '=', 'like', '!=', '-', '+', 'between', 'and', 'or', 'not', 'in', ')', 'by', 'distinct', '/']
sql_ops_no_space = ['count', 'avg', 'sum', 'max', 'min', '(', '!', 'desc', 'asc']
sql_marks = [',', 'as']

def lower_(
    word: str,
) -> str:
    if '"' in word or "'" in word:
        return word
    else:
        return word.lower()

def normalize_alias(
    sql: str,
    table_names: List[str],
) -> str:
    alias_format = 'T{count}'
    count = 1
    for tab in table_names+['DERIVED_FIELD', 'DERIVED_TABLE']:
        tab = tab.upper()
        for idx in ['0','1','2','3','4','5','6']:
            old_alias = tab+'alias'+idx
            if old_alias in sql:
                new_alias = alias_format.format(count=count)
                sql = sql.replace(old_alias, new_alias)
                count += 1
    return sql
def tok_process(
    toks: list,
) -> list:
    processed_tok_list = []
    i = 0
    while i < len(toks):
        if toks[i] == "``" and toks[i+2] == "''":
            temp = f'"{toks[i+1]}"'
            processed_tok_list.append(temp)
            i += 3
            continue
        else:
            processed_tok_list.append(toks[i])
            i += 1
    return [lower_(x) for x in processed_tok_list]

def geoquery_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
) -> str:
    return prefix + question.strip() + " | " + serialized_schema.strip()


def geoquery_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)

def serialize_schema(
    question: str,
    normalize_query: bool = True,
    schema_serialization_type: str = "verbose",
    schema_serialization_with_db_id: bool = True,
    schema_serialization_with_prompt: str = "",
    schema_serialization_with_db_content: bool = False,
) -> str:
    if schema_serialization_type == "verbose":
        #([col] needs to refer to columns, [val] needs to refer to words in the question.)
        db_id_str = "database: {db_id}. "
        table_sep = ". "
        table_str = "table: {table}. columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError
    
    table_dict = {
        "border_info": {"state_name": ['alabama', 'alabama', 'alabama', 'alabama', 'arizona', 'arizona', 'arizona', 'arizona', 'arizona', 'arkansas', 'arkansas', 'arkansas', 'arkansas', 'arkansas', 'arkansas', 'california', 'california', 'california', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'connecticut', 'connecticut', 'connecticut', 'delaware', 'delaware', 'delaware', 'district of columbia', 'district of columbia', 'florida', 'florida', 'georgia', 'georgia', 'georgia', 'georgia', 'georgia', 'idaho', 'idaho', 'idaho', 'idaho', 'idaho', 'idaho', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'indiana', 'indiana', 'indiana', 'indiana', 'iowa', 'iowa', 'iowa', 'iowa', 'iowa', 'iowa', 'kansas', 'kansas', 'kansas', 'kansas', 'kentucky', 'kentucky', 'kentucky', 'kentucky', 'kentucky', 'kentucky', 'kentucky', 'louisiana', 'louisiana', 'louisiana', 'maine', 'maryland', 'maryland', 'maryland', 'maryland', 'maryland', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'michigan', 'michigan', 'michigan', 'minnesota', 'minnesota', 'minnesota', 'minnesota', 'mississippi', 'mississippi', 'mississippi', 'mississippi', 'missouri', 'missouri', 'missouri', 'missouri', 'missouri', 'missouri', 'missouri', 'missouri', 'montana', 'montana', 'montana', 'montana', 'nebraska', 'nebraska', 'nebraska', 'nebraska', 'nebraska', 'nebraska', 'nevada', 'nevada', 'nevada', 'nevada', 'nevada', 'new hampshire', 'new hampshire', 'new hampshire', 'new jersey', 'new jersey', 'new jersey', 'new mexico', 'new mexico', 'new mexico', 'new mexico', 'new mexico', 'new york', 'new york', 'new york', 'new york', 'new york', 'north carolina', 'north carolina', 'north carolina', 'north carolina', 'north dakota', 'north dakota', 'north dakota', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'oklahoma', 'oklahoma', 'oklahoma', 'oklahoma', 'oklahoma', 'oklahoma', 'oregon', 'oregon', 'oregon', 'oregon', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'rhode island', 'rhode island', 'south carolina', 'south carolina', 'south dakota', 'south dakota', 'south dakota', 'south dakota', 'south dakota', 'south dakota', 'tennessee', 'tennessee', 'tennessee', 'tennessee', 'tennessee', 'tennessee', 'tennessee', 'tennessee', 'texas', 'texas', 'texas', 'texas', 'utah', 'utah', 'utah', 'utah', 'utah', 'utah', 'vermont', 'vermont', 'vermont', 'virginia', 'virginia', 'virginia', 'virginia', 'virginia', 'virginia', 'washington', 'washington', 'west virginia', 'west virginia', 'west virginia', 'west virginia', 'west virginia', 'wisconsin', 'wisconsin', 'wisconsin', 'wisconsin', 'wyoming', 'wyoming', 'wyoming', 'wyoming', 'wyoming', 'wyoming'], 
                        "border": ['tennessee', 'georgia', 'florida', 'mississippi', 'utah', 'colorado', 'new mexico', 'california', 'nevada', 'missouri', 'tennessee', 'mississippi', 'louisiana', 'texas', 'oklahoma', 'oregon', 'nevada', 'arizona', 'nebraska', 'kansas', 'oklahoma', 'new mexico', 'arizona', 'utah', 'wyoming', 'massachusetts', 'rhode island', 'new york', 'pennsylvania', 'new jersey', 'maryland', 'maryland', 'virginia', 'georgia', 'alabama', 'north carolina', 'south carolina', 'florida', 'alabama', 'tennessee', 'montana', 'wyoming', 'utah', 'nevada', 'oregon', 'washington', 'wisconsin', 'indiana', 'kentucky', 'missouri', 'iowa', 'michigan', 'ohio', 'kentucky', 'illinois', 'minnesota', 'wisconsin', 'illinois', 'missouri', 'nebraska', 'south dakota', 'nebraska', 'missouri', 'oklahoma', 'colorado', 'indiana', 'ohio', 'west virginia', 'virginia', 'tennessee', 'missouri', 'illinois', 'arkansas', 'mississippi', 'texas', 'new hampshire', 'pennsylvania', 'delaware', 'virginia', 'district of columbia', 'west virginia', 'new hampshire', 'rhode island', 'connecticut', 'new york', 'vermont', 'ohio', 'indiana', 'wisconsin', 'wisconsin', 'iowa', 'south dakota', 'north dakota', 'tennessee', 'alabama', 'louisiana', 'arkansas', 'iowa', 'illinois', 'kentucky', 'tennessee', 'arkansas', 'oklahoma', 'kansas', 'nebraska', 'north dakota', 'south dakota', 'wyoming', 'idaho', 'south dakota', 'iowa', 'missouri', 'kansas', 'colorado', 'wyoming', 'idaho', 'utah', 'arizona', 'california', 'oregon', 'maine', 'massachusetts', 'vermont', 'new york', 'delaware', 'pennsylvania', 'colorado', 'oklahoma', 'texas', 'arizona', 'utah', 'vermont', 'massachusetts', 'connecticut', 'new jersey', 'pennsylvania', 'virginia', 'south carolina', 'georgia', 'tennessee', 'minnesota', 'south dakota', 'montana', 'michigan', 'pennsylvania', 'west virginia', 'kentucky', 'indiana', 'kansas', 'missouri', 'arkansas', 'texas', 'new mexico', 'colorado', 'washington', 'idaho', 'nevada', 'california', 'new york', 'new jersey', 'delaware', 'maryland', 'west virginia', 'ohio', 'massachusetts', 'connecticut', 'north carolina', 'georgia', 'north dakota', 'minnesota', 'iowa', 'nebraska', 'wyoming', 'montana', 'kentucky', 'virginia', 'north carolina', 'georgia', 'alabama', 'mississippi', 'arkansas', 'missouri', 'oklahoma', 'arkansas', 'louisiana', 'new mexico', 'wyoming', 'colorado', 'new mexico', 'arizona', 'nevada', 'idaho', 'new hampshire', 'massachusetts', 'new york', 'maryland', 'district of columbia', 'north carolina', 'tennessee', 'kentucky', 'west virginia', 'idaho', 'oregon', 'pennsylvania', 'maryland', 'virginia', 'kentucky', 'ohio', 'michigan', 'illinois', 'iowa', 'minnesota', 'montana', 'south dakota', 'nebraska', 'colorado', 'utah', 'idaho']},
        "city": {"city_name": ['birmingham', 'mobile', 'montgomery', 'huntsville', 'tuscaloosa', 'anchorage', 'phoenix', 'tucson', 'mesa', 'tempe', 'glendale', 'scottsdale', 'little rock', 'fort smith', 'north little rock', 'los angeles', 'san diego', 'san francisco', 'san jose', 'long beach', 'oakland', 'sacramento', 'anaheim', 'fresno', 'santa ana', 'riverside', 'huntington beach', 'stockton', 'glendale', 'fremont', 'torrance', 'garden grove', 'san bernardino', 'pasadena', 'east los angeles', 'oxnard', 'modesto', 'sunnyvale', 'bakersfield', 'concord', 'berkeley', 'fullerton', 'inglewood', 'hayward', 'pomona', 'orange', 'ontario', 'santa monica', 'santa clara', 'citrus heights', 'norwalk', 'burbank', 'chula vista', 'santa rosa', 'downey', 'costa mesa', 'compton', 'carson', 'salinas', 'west covina', 'vallejo', 'el monte', 'daly city', 'thousand oaks', 'san mateo', 'simi valley', 'oceanside', 'richmond', 'lakewood', 'santa barbara', 'el cajon', 'ventura', 'westminster', 'whittier', 'south gate', 'alhambra', 'buena park', 'san leandro', 'alameda', 'newport beach', 'escondido', 'irvine', 'mountain view', 'fairfield', 'redondo beach', 'scotts valley', 'denver', 'colorado springs', 'aurora', 'lakewood', 'pueblo', 'arvada', 'boulder', 'fort collins', 'bridgeport', 'hartford', 'new haven', 'waterbury', 'stamford', 'norwalk', 'new britain', 'west hartford', 'danbury', 'greenwich', 'bristol', 'meriden', 'wilmington', 'washington', 'jacksonville', 'miami', 'tampa', 'st. petersburg', 'fort lauderdale', 'orlando', 'hollywood', 'miami beach', 'clearwater', 'tallahassee', 'gainesville', 'kendall', 'west palm beach', 'largo', 'pensacola', 'atlanta', 'columbus', 'savannah', 'macon', 'albany', 'honolulu', 'ewa', 'koolaupoko', 'boise', 'chicago', 'rockford', 'peoria', 'springfield', 'decatur', 'aurora', 'joliet', 'evanston', 'waukegan', 'arlington heights', 'elgin', 'cicero', 'oak lawn', 'skokie', 'champaign', 'indianapolis', 'fort wayne', 'gary', 'evansville', 'south bend', 'hammond', 'muncie', 'anderson', 'terre haute', 'des moines', 'cedar rapids', 'davenport', 'sioux city', 'waterloo', 'dubuque', 'wichita', 'kansas city', 'topeka', 'overland park', 'louisville', 'lexington', 'new orleans', 'baton rouge', 'shreveport', 'metairie', 'lafayette', 'lake charles', 'kenner', 'monroe', 'portland', 'baltimore', 'silver spring', 'dundalk', 'bethesda', 'boston', 'worcester', 'springfield', 'new bedford', 'cambridge', 'brockton', 'fall river', 'lowell', 'quincy', 'newton', 'lynn', 'somerville', 'framingham', 'lawrence', 'waltham', 'medford', 'detroit', 'grand rapids', 'warren', 'flint', 'lansing', 'sterling heights', 'ann arbor', 'livonia', 'dearborn', 'westland', 'kalamazoo', 'taylor', 'saginaw', 'pontiac', 'st. clair shores', 'southfield', 'clinton', 'royal oak', 'dearborn heights', 'troy', 'waterford', 'wyoming', 'redford', 'farmington hills', 'minneapolis', 'st. paul', 'duluth', 'bloomington', 'rochester', 'jackson', 'st. louis', 'kansas city', 'springfield', 'independence', 'st. joseph', 'columbia', 'billings', 'great falls', 'omaha', 'lincoln', 'las vegas', 'reno', 'manchester', 'nashua', 'newark', 'jersey city', 'paterson', 'elizabeth', 'trenton', 'woodbridge', 'camden', 'east orange', 'clifton', 'edison', 'cherry hill', 'bayonne', 'middletown', 'irvington', 'albuquerque', 'new york', 'buffalo', 'rochester', 'yonkers', 'syracuse', 'albany', 'cheektowaga', 'utica', 'niagara falls', 'new rochelle', 'schenectady', 'mount vernon', 'irondequoit', 'levittown', 'charlotte', 'greensboro', 'raleigh', 'winston-salem', 'durham', 'high point', 'fayetteville', 'fargo', 'cleveland', 'columbus', 'cincinnati', 'toledo', 'akron', 'dayton', 'youngstown', 'canton', 'parma', 'lorain', 'springfield', 'hamilton', 'lakewood', 'kettering', 'euclid', 'elyria', 'oklahoma city', 'tulsa', 'lawton', 'norman', 'portland', 'eugene', 'salem', 'philadelphia', 'pittsburgh', 'erie', 'allentown', 'scranton', 'upper darby', 'reading', 'bethlehem', 'lower merion', 'abingdon', 'bristol township', 'penn hills', 'altoona', 'providence', 'warwick', 'cranston', 'pawtucket', 'columbia', 'charleston', 'north charleston', 'greenville', 'sioux falls', 'memphis', 'nashville', 'knoxville', 'chattanooga', 'houston', 'dallas', 'san antonio', 'el paso', 'fort worth', 'austin', 'corpus christi', 'lubbock', 'arlington', 'amarillo', 'garland', 'beaumont', 'pasadena', 'irving', 'waco', 'abilene', 'wichita falls', 'laredo', 'odessa', 'brownsville', 'san angelo', 'richardson', 'plano', 'grand prairie', 'midland', 'tyler', 'mesquite', 'mcallen', 'longview', 'port arthur', 'salt lake city', 'provo', 'west valley', 'ogden', 'norfolk', 'virginia beach', 'richmond', 'arlington', 'newport news', 'hampton', 'chesapeake', 'portsmouth', 'alexandria', 'roanoke', 'lynchburg', 'seattle', 'spokane', 'tacoma', 'bellevue', 'charleston', 'huntington', 'milwaukee', 'madison', 'green bay', 'racine', 'kenosha', 'west allis', 'appleton', 'casper'],
                 "population": [284413, 200452, 177857, 142513, 75143, 174431, 789704, 330537, 152453, 106919, 96988, 88622, 158915, 71384, 64388, 2966850, 875538, 678974, 629442, 361334, 339337, 275741, 219311, 218202, 203713, 170876, 170505, 149779, 139060, 131945, 131497, 123351, 118794, 118072, 110017, 108195, 106963, 106618, 105611, 103763, 103328, 102246, 94162, 93585, 92742, 91450, 88820, 88314, 87700, 85911, 84901, 84625, 83927, 83205, 82602, 82291, 81230, 81221, 80479, 80292, 80188, 79494, 78519, 77797, 77640, 77500, 76698, 74676, 74654, 74542, 73892, 73774, 71133, 68558, 66784, 64767, 64165, 63952, 63852, 63475, 62480, 62134, 58655, 58099, 57102, 6037, 492365, 215150, 158588, 113808, 101686, 84576, 76685, 64632, 142546, 136392, 126089, 103266, 102466, 77767, 73840, 61301, 60470, 59578, 57370, 57118, 70195, 638333, 540920, 346865, 271523, 238647, 153256, 128394, 117188, 96298, 85450, 81548, 81371, 73758, 62530, 58977, 57619, 425022, 169441, 141654, 116860, 74425, 762874, 190037, 109373, 102249, 3005172, 139712, 124160, 100054, 93939, 81293, 77956, 73706, 67653, 66116, 63668, 61232, 60590, 60278, 58267, 700807, 172196, 151968, 130496, 109727, 93714, 77216, 64695, 61125, 191003, 110243, 103254, 82003, 75985, 62321, 279212, 161148, 118690, 81784, 298451, 204165, 557515, 219419, 205820, 164160, 80584, 75051, 66382, 57597, 61572, 786775, 72893, 71293, 63022, 562994, 161799, 152319, 98478, 95322, 95172, 92574, 92418, 84743, 83622, 78471, 77372, 65113, 63175, 58200, 58076, 1203339, 181843, 161134, 159611, 130414, 108999, 107969, 104814, 90660, 84603, 79722, 77568, 77508, 76715, 76210, 75568, 72400, 70893, 67706, 67102, 64250, 59616, 58441, 58056, 370951, 270230, 92811, 81831, 57906, 202895, 453085, 448159, 133116, 111797, 76691, 62061, 66842, 56725, 314255, 171932, 164674, 100756, 90936, 67865, 329248, 223532, 137970, 106201, 92124, 90074, 84910, 77878, 74388, 70193, 68785, 65047, 61615, 61493, 331767, 7071639, 357870, 241741, 195351, 170105, 101727, 92145, 75632, 71384, 70794, 67972, 66713, 57648, 57045, 314447, 155642, 149771, 131885, 100538, 64107, 59507, 61308, 573822, 564871, 385457, 354635, 237177, 203371, 115436, 93077, 92548, 75416, 72563, 63189, 61963, 61186, 59999, 57504, 403213, 360919, 80054, 68020, 366383, 105664, 89233, 1688210, 423938, 119123, 103758, 88117, 84054, 78686, 70419, 59651, 59084, 58733, 57632, 57078, 156804, 87123, 71992, 71204, 101229, 69855, 62504, 58242, 81343, 646356, 455651, 175030, 169728, 1595138, 904078, 785880, 425259, 385164, 345496, 231999, 173979, 160123, 149230, 138857, 118102, 112560, 109943, 101261, 98315, 94201, 91449, 90027, 84997, 73240, 72496, 72331, 71462, 70525, 70508, 67053, 67042, 62762, 61195, 163034, 74111, 72299, 64407, 266979, 262199, 219214, 152599, 144903, 122617, 114226, 104577, 103217, 100427, 66743, 493846, 171300, 158501, 73903, 63968, 63684, 636212, 170616, 87899, 85725, 77685, 63982, 58913, 51016],
                 "country_name": ['usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa'],
                 "state_name": ['alabama', 'alabama', 'alabama', 'alabama', 'alabama', 'alaska', 'arizona', 'arizona', 'arizona', 'arizona', 'arizona', 'arizona', 'arkansas', 'arkansas', 'arkansas', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'california', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'delaware', 'district of columbia', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'florida', 'georgia', 'georgia', 'georgia', 'georgia', 'georgia', 'hawaii', 'hawaii', 'hawaii', 'idaho', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'illinois', 'indiana', 'indiana', 'indiana', 'indiana', 'indiana', 'indiana', 'indiana', 'indiana', 'indiana', 'iowa', 'iowa', 'iowa', 'iowa', 'iowa', 'iowa', 'kansas', 'kansas', 'kansas', 'kansas', 'kentucky', 'kentucky', 'louisiana', 'louisiana', 'louisiana', 'louisiana', 'louisiana', 'louisiana', 'louisiana', 'louisiana', 'maine', 'maryland', 'maryland', 'maryland', 'maryland', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'massachusetts', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'minnesota', 'minnesota', 'minnesota', 'minnesota', 'minnesota', 'mississippi', 'missouri', 'missouri', 'missouri', 'missouri', 'missouri', 'missouri', 'montana', 'montana', 'nebraska', 'nebraska', 'nevada', 'nevada', 'new hampshire', 'new hampshire', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new jersey', 'new mexico', 'new york', 'new york', 'new york', 'new york', 'new york', 'new york', 'new york', 'new york', 'new york', 'new york', 'new york', 'new york', 'new york', 'new york', 'north carolina', 'north carolina', 'north carolina', 'north carolina', 'north carolina', 'north carolina', 'north carolina', 'north dakota', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'oklahoma', 'oklahoma', 'oklahoma', 'oklahoma', 'oregon', 'oregon', 'oregon', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'pennsylvania', 'rhode island', 'rhode island', 'rhode island', 'rhode island', 'south carolina', 'south carolina', 'south carolina', 'south carolina', 'south dakota', 'tennessee', 'tennessee', 'tennessee', 'tennessee', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'texas', 'utah', 'utah', 'utah', 'utah', 'virginia', 'virginia', 'virginia', 'virginia', 'virginia', 'virginia', 'virginia', 'virginia', 'virginia', 'virginia', 'virginia', 'washington', 'washington', 'washington', 'washington', 'west virginia', 'west virginia', 'wisconsin', 'wisconsin', 'wisconsin', 'wisconsin', 'wisconsin', 'wisconsin', 'wisconsin', 'wyoming']},
        "highlow": {"state_name": ['alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'district of columbia', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming'],
                    "highest_elevation": ['734', '6194', '3851', '839', '4418', '4399', '725', '135', '125', '105', '1458', '4205', '3859', '376', '383', '511', '1231', '1263', '163', '1606', '1024', '1064', '604', '701', '246', '540', '3901', '1654', '4005', '1917', '550', '4011', '1629', '2037', '1069', '472', '1516', '3424', '979', '247', '1085', '2207', '2025', '2667', '4123', '1339', '1746', '4392', '1482', '595', '4202'],
                    "lowest_point": ['gulf of mexico', 'pacific ocean', 'colorado river', 'ouachita river', 'death valley', 'arkansas river', 'long island sound', 'atlantic ocean', 'potomac river', 'atlantic ocean', 'atlantic ocean', 'pacific ocean', 'snake river', 'mississippi river', 'ohio river', 'mississippi river', 'verdigris river', 'mississippi river', 'new orleans', 'atlantic ocean', 'atlantic ocean', 'atlantic ocean', 'lake erie', 'lake superior', 'gulf of mexico', 'st. francis river', 'kootenai river', 'southeast corner', 'colorado river', 'atlantic ocean', 'atlantic ocean', 'red bluff reservoir', 'atlantic ocean', 'atlantic ocean', 'red river', 'ohio river', 'little river', 'pacific ocean', 'delaware river', 'atlantic ocean', 'atlantic ocean', 'big stone lake', 'mississippi river', 'gulf of mexico', 'beaver dam creek', 'lake champlain', 'atlantic ocean', 'pacific ocean', 'potomac river', 'lake michigan', 'belle fourche river'],
                    "highest_point": ['cheaha mountain', 'mount mckinley', 'humphreys peak', 'magazine mountain', 'mount whitney', 'mount elbert', 'mount frissell', 'centerville', 'tenleytown', 'walton county', 'brasstown bald', 'mauna kea', 'borah peak', 'charles mound', 'franklin township', 'ocheyedan mound', 'mount sunflower', 'black mountain', 'driskill mountain', 'mount katahdin', 'backbone mountain', 'mount greylock', 'mount curwood', 'eagle mountain', 'woodall mountain', 'taum sauk mountain', 'granite peak', 'johnson township', 'boundary peak', 'mount washington', 'high point', 'wheeler peak', 'mount marcy', 'mount mitchell', 'white butte', 'campbell hill', 'black mesa', 'mount hood', 'mount davis', 'jerimoth hill', 'sassafras mountain', 'harney peak', 'clingmans dome', 'guadalupe peak', 'kings peak', 'mount mansfield', 'mount rogers', 'mount rainier', 'spruce knob', 'timms hill', 'gannett peak'],
                    "lowest_elevation": ['0', '0', '21', '17', '-85', '1021', '0', '0', '0', '0', '0', '0', '216', '85', '98', '146', '207', '78', '-1', '0', '0', '0', '174', '183', '0', '70', '549', '256', '143', '0', '0', '859', '0', '0', '229', '132', '87', '0', '0', '0', '0', '284', '55', '0', '610', '29', '0', '0', '73', '177', '945']},
        "lake": {"lake_name": ['iliamna', 'becharof', 'teshekpuk', 'naknek', 'salton sea', 'tahoe', 'okeechobee', 'michigan', 'michigan', 'pontchartrain', 'superior', 'huron', 'michigan', 'erie', 'st. clair', 'superior', 'lake of the woods', 'red', 'rainy', 'mille lacs', 'flathead', 'tahoe', 'erie', 'ontario', 'champlain', 'erie', 'erie', 'great salt lake', 'champlain', 'superior', 'michigan', 'winnebago'],
                 "area": [2675, 1186, 816, 630, 932, 497, 1810, 58016, 58016, 1632, 82362, 59570, 58016, 25667, 1119, 82362, 4391, 1169, 932, 536, 510, 497, 25667, 19684, 1114, 25667, 25667, 5180, 1114, 82362, 58016, 557],
                 "country_name": ['usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa'],
                 "state_name": ['alaska', 'alaska', 'alaska', 'alaska', 'california', 'california', 'florida', 'illinois', 'indiana', 'louisiana', 'michigan', 'michigan', 'michigan', 'michigan', 'michigan', 'minnesota', 'minnesota', 'minnesota', 'minnesota', 'minnesota', 'montana', 'nevada', 'new york', 'new york', 'new york', 'ohio', 'pennsylvania', 'utah', 'vermont', 'wisconsin', 'wisconsin', 'wisconsin']},
        "mountain": {"mountain_name": ['mckinley', 'st. elias', 'foraker', 'bona', 'blackburn', 'kennedy', 'sanford', 'south buttress', 'vancouver', 'churchill', 'fairweather', 'hubbard', 'bear', 'east buttress', 'hunter', 'alverstone', 'browne tower', 'wrangell', 'whitney', 'williamson', 'white', 'north palisade', 'shasta', 'sill', 'elbert', 'massive', 'harvard', 'bianca', 'la plata', 'uncompahgre', 'crestone', 'lincoln', 'grays', 'antero', 'torreys', 'castle', 'quandary', 'evans', 'longs', 'wilson', 'shavano', 'belford', 'princeton', 'crestone needle', 'yale', 'bross', 'kit carson', 'el diente', 'maroon', 'rainier'],
                     "mountain_altitude": [6194, 5489, 5304, 5044, 4996, 4964, 4949, 4842, 4785, 4766, 4663, 4577, 4520, 4490, 4442, 4439, 4429, 4317, 4418, 4382, 4342, 4341, 4317, 4317, 4399, 4396, 4395, 4372, 4370, 4361, 4357, 4354, 4349, 4349, 4349, 4348, 4348, 4348, 4345, 4342, 4337, 4327, 4327, 4327, 4327, 4320, 4317, 4316, 4315, 4392],
                     "country_name": ['usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa'],
                     "state_name": ['alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'alaska', 'california', 'california', 'california', 'california', 'california', 'california', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'washington']},
        "river": {"river_name": ['mississippi', 'mississippi', 'mississippi', 'mississippi', 'mississippi', 'mississippi', 'mississippi', 'mississippi', 'mississippi', 'mississippi', 'mississippi', 'missouri', 'missouri', 'missouri', 'missouri', 'missouri', 'missouri', 'missouri', 'colorado', 'colorado', 'colorado', 'colorado', 'colorado', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'ohio', 'red', 'red', 'red', 'red', 'red', 'red', 'arkansas', 'arkansas', 'arkansas', 'arkansas', 'canadian', 'canadian', 'canadian', 'canadian', 'connecticut', 'connecticut', 'connecticut', 'connecticut', 'delaware', 'delaware', 'delaware', 'delaware', 'little missouri', 'little missouri', 'little missouri', 'little missouri', 'snake', 'snake', 'snake', 'snake', 'snake', 'chattahoochee', 'chattahoochee', 'chattahoochee', 'cimarron', 'cimarron', 'cimarron', 'green', 'green', 'green', 'green', 'north platte', 'north platte', 'north platte', 'potomac', 'potomac', 'potomac', 'potomac', 'republican', 'republican', 'republican', 'rio grande', 'rio grande', 'rio grande', 'san juan', 'san juan', 'san juan', 'san juan', 'tennessee', 'tennessee', 'tennessee', 'tennessee', 'wabash', 'wabash', 'wabash', 'yellowstone', 'yellowstone', 'yellowstone', 'allegheny', 'allegheny', 'allegheny', 'bighorn', 'bighorn', 'cheyenne', 'cheyenne', 'clark fork', 'clark fork', 'columbia', 'columbia', 'cumberland', 'cumberland', 'cumberland', 'dakota', 'dakota', 'gila', 'gila', 'hudson', 'hudson', 'neosho', 'neosho', 'niobrara', 'niobrara', 'ouachita', 'ouachita', 'pearl', 'pearl', 'pecos', 'pecos', 'powder', 'powder', 'roanoke', 'roanoke', 'rock', 'rock', 'smoky hill', 'smoky hill', 'south platte', 'south platte', 'st. francis', 'st. francis', 'tombigbee', 'tombigbee', 'washita', 'washita', 'wateree catawba', 'wateree catawba', 'white', 'white', 'white'],
                  "length": [3778, 3778, 3778, 3778, 3778, 3778, 3778, 3778, 3778, 3778, 3778, 3968, 3968, 3968, 3968, 3968, 3968, 3968, 2333, 2333, 2333, 2333, 2333, 1569, 1569, 1569, 1569, 1569, 1569, 1569, 1638, 1638, 1638, 1638, 1638, 1638, 2333, 2333, 2333, 2333, 1458, 1458, 1458, 1458, 655, 655, 655, 655, 451, 451, 451, 451, 901, 901, 901, 901, 1670, 1670, 1670, 1670, 1670, 702, 702, 702, 965, 965, 965, 1175, 1175, 1175, 1175, 1094, 1094, 1094, 462, 462, 462, 462, 679, 679, 679, 3033, 3033, 3033, 579, 579, 579, 579, 1049, 1049, 1049, 1049, 764, 764, 764, 1080, 1080, 1080, 523, 523, 523, 541, 541, 848, 848, 483, 483, 1953, 1953, 1105, 1105, 1105, 1142, 1142, 805, 805, 492, 492, 740, 740, 693, 693, 973, 973, 788, 788, 805, 805, 603, 603, 660, 660, 459, 459, 869, 869, 682, 682, 684, 684, 658, 658, 805, 805, 636, 636, 1110, 1110, 1110],
                  "country_name": ['usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa'],
                  "traverse": ['minnesota', 'wisconsin', 'iowa', 'illinois', 'missouri', 'kentucky', 'tennessee', 'arkansas', 'mississippi', 'louisiana', 'louisiana', 'montana', 'north dakota', 'south dakota', 'iowa', 'nebraska', 'missouri', 'missouri', 'colorado', 'utah', 'arizona', 'nevada', 'california', 'pennsylvania', 'west virginia', 'kentucky', 'indiana', 'illinois', 'illinois', 'ohio', 'new mexico', 'texas', 'oklahoma', 'arkansas', 'arkansas', 'louisiana', 'colorado', 'kansas', 'oklahoma', 'arkansas', 'colorado', 'new mexico', 'texas', 'oklahoma', 'new hampshire', 'vermont', 'massachusetts', 'connecticut', 'new york', 'pennsylvania', 'new jersey', 'delaware', 'wyoming', 'montana', 'south dakota', 'north dakota', 'wyoming', 'idaho', 'oregon', 'washington', 'washington', 'georgia', 'georgia', 'florida', 'new mexico', 'kansas', 'oklahoma', 'wyoming', 'utah', 'colorado', 'utah', 'colorado', 'wyoming', 'nebraska', 'west virginia', 'maryland', 'virginia', 'district of columbia', 'colorado', 'nebraska', 'kansas', 'colorado', 'new mexico', 'texas', 'colorado', 'new mexico', 'colorado', 'utah', 'tennessee', 'alabama', 'tennessee', 'kentucky', 'ohio', 'indiana', 'illinois', 'wyoming', 'montana', 'north dakota', 'pennsylvania', 'new york', 'pennsylvania', 'wyoming', 'montana', 'wyoming', 'north dakota', 'montana', 'idaho', 'washington', 'oregon', 'kentucky', 'tennessee', 'kentucky', 'north dakota', 'south dakota', 'new mexico', 'arizona', 'new york', 'new jersey', 'kansas', 'oklahoma', 'wyoming', 'nebraska', 'arkansas', 'louisiana', 'michigan', 'louisiana', 'new mexico', 'texas', 'wyoming', 'montana', 'virginia', 'north carolina', 'wisconsin', 'illinois', 'colorado', 'kansas', 'colorado', 'nebraska', 'missouri', 'arkansas', 'mississippi', 'alabama', 'texas', 'oklahoma', 'north carolina', 'south carolina', 'arkansas', 'missouri', 'arkansas']},
        "state": {"state_name": ['alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'district of columbia', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming'],
                  "population": [3894000, 401800, 2718000, 2286000, 23670000, 2889000, 3107000, 594000, 638000, 9746000, 5463000, 964000, 944000, 11400000, 5490000, 2913000, 2364000, 2364000, 4206000, 1125000, 4217000, 5737000, 9262000, 4076000, 2520000, 4916000, 786700, 1569000, 800500, 920600, 7365000, 1303000, 17558000, 5882000, 652700, 10800000, 3025000, 2633000, 11863000, 947200, 3121800, 690767, 4591000, 14229000, 1461000, 511500, 5346800, 4113200, 1950000, 4700000, 469557],
                  "area": [51700, 591000, 114000, 53200, 158000, 104000, 5020, 2044, 1100, 68664, 58900, 6471, 83000, 56300, 36200, 56300, 82300, 82300, 47700, 33265, 10460, 8284, 58500, 84400, 47700, 69700, 147000, 77300, 110500, 9279, 7787, 121600, 49100, 52670, 70700, 41300, 69950, 97073, 45308, 1212, 31113, 77116, 42140, 266807, 84900, 9614, 40760, 68139, 24200, 56153, 97809],
                  "country_name": ['usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa'],
                  "capital": ['montgomery', 'juneau', 'phoenix', 'little rock', 'sacramento', 'denver', 'hartford', 'dover', 'washington', 'tallahassee', 'atlanta', 'honolulu', 'boise', 'springfield', 'indianapolis', 'des moines', 'topeka', 'frankfort', 'baton rouge', 'augusta', 'annapolis', 'boston', 'lansing', 'st. paul', 'jackson', 'jefferson city', 'helena', 'lincoln', 'carson city', 'concord', 'trenton', 'santa fe', 'albany', 'raleigh', 'bismarck', 'columbus', 'oklahoma city', 'salem', 'harrisburg', 'providence', 'columbia', 'pierre', 'nashville', 'austin', 'salt lake city', 'montpelier', 'richmond', 'olympia', 'charleston', 'madison', 'cheyenne'],
                  "density": [75.31914893617021, 0.6798646362098139, 23.842105263157894, 42.96992481203007, 149.81012658227849, 27.778846153846153, 618.9243027888447, 290.60665362035223, 580, 141.9375509728533, 92.75042444821732, 148.97233812393756, 11.373493975903614, 202.4866785079929, 151.65745856353593, 51.740674955595026, 28.724179829890645, 28.724179829890645, 88.17610062893081, 33.81932962573275, 403.1548757170172, 692.5398358281024, 158.32478632478632, 48.29383886255924, 52.83018867924528, 70.53084648493544, 5.351700680272109, 20.297542043984475, 7.244343891402715, 99.21327729281172, 945.8071144214717, 10.71546052631579, 357.5967413441955, 111.67647617239415, 9.231966053748232, 261.50121065375305, 43.24517512508935, 27.12391705211542, 261.8301403725611, 781.5181518151816, 100.3374795101726, 8.957505576015354, 108.94636924537257, 53.33068472716233, 17.208480565371026, 53.203661327231124, 131.1776251226693, 60.36484245439469, 80.57851239669421, 83.69989136822609, 4.8007545317915525]}
    }
    tables = []
    for table_name in table_dict:
        table_str = f"table: {table_name}. columns: "
        for col_name in table_dict[table_name]:
            pick_val = None
            for val in list(set(table_dict[table_name][col_name])):
                if str(val).lower() in question.lower():
                    pick_val = val.lower()
                    break
            if pick_val == None:
                pick_val = random.choice(list(set(table_dict[table_name][col_name])))
            table_str += f"{col_name} ({pick_val}),"
        table_str = table_str[:-1]
        tables.append(table_str)

    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id='geo') + table_sep.join(tables)
    else:
        serialized_schema = 'database: ' + table_sep.join(tables)

    return serialized_schema

def geoquery_add_serialized_schema(ex: dict, mode: str, data_training_args: DataTrainingArguments) -> dict:
    serialized_schema = serialize_schema(
        question=ex["question"],
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_prompt=data_training_args.schema_serialization_with_prompt,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        normalize_query=data_training_args.normalize_query,
    )
    return {"serialized_schema": serialized_schema}


def geoquery_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    mode: Optional[str],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else "question: "
    if data_training_args.use_decomposition:
        inputs = []
        targets = []
        if data_training_args.stage == 'content':
            eval_format_list = []
            with open(data_training_args.structure_path) as f:
                info = json.load(f)
                for item in info:
                    eval_format_list.append(item['prediction'])
            print(f"load {len(eval_format_list)} eval_formats")
        count = 0
        for question, serialized_schema, sql, db_column_names, table_names_ in zip(batch["question"], batch["serialized_schema"], batch["query"], batch["db_column_names"], batch["db_table_names"]):
            # sql = normalize_alias(sql, table_names_)
            input_str = geoquery_get_input(question=question + '?', serialized_schema=serialized_schema, prefix=prefix)
            column_names = [x.lower() for x in db_column_names['column_name']]
            query_toks = sql.split()
            for idx, tok in enumerate(query_toks):
                if len(tok) > 1:
                    if tok[-1] in ['(', ')']:
                        query_toks.insert(idx+1, tok[-1])
                        query_toks[idx] = tok[:-1]
                    if tok[0] in ['(', ')']:
                        query_toks.insert(idx, tok[0])
                        query_toks[idx+1] = tok[1:]

            query_tok_list = tok_process(query_toks)
            sub_query_format_list = []
            sub_query_list = []
            select_from_record = []
            sub_query_format = ''
            sub_query = ''
            content_label = ''
            last_tok = None
            select_num = 0
            left_bracket = 0
            right_bracket = 0
            if query_tok_list[-1] == ';':
                query_tok_list = query_tok_list[:-1]
            for idx, query_tok in enumerate(query_tok_list):
                if query_tok in sql_clauses:
                    if query_tok == 'select':
                        select_num += 1
                    elif (query_tok == 'group' or query_tok == 'order') and query_tok_list[idx+1] != 'by':
                        if query_tok in column_names:
                            if idx + 1 == len(query_tok_list) or query_tok_list[idx+1] in [',', ')']:
                                sub_query_format += '[col]'
                            else:
                                sub_query_format += '[col] '
                            sub_query += query_tok + ' '
                            content_label += '[col] ' + query_tok + ' '
                        else:
                            print("error:", query_tok)
                        continue

                    if sub_query_format != '' and sub_query != '':
                        if 'select' in sub_query_format:
                            select_from_record.append(1)
                        elif 'from' in sub_query_format:
                            select_from_record.append(2)
                        else:
                            select_from_record.append(0)
                        sub_query_format_list.append(sub_query_format)
                        sub_query_list.append(sub_query)
                        sub_query_format = ''
                        sub_query = ''
                    if query_tok == 'from':
                        if query_tok_list[idx+1] != '(':
                            sub_query_format += 'from [tab]'
                            content_label += '[tab] '
                        else:
                            sub_query_format += 'from'
                    else:
                        sub_query_format += query_tok + ' '
                    last_tok = 'sql_clauses'
                    sub_query += query_tok + ' '
                elif sub_query_format == 'from [tab]':
                    last_tok = 'from [tab]'
                    sub_query += query_tok + ' '
                    if query_tok not in [')', '(']:
                        content_label += query_tok + ' '
                    continue
                elif query_tok in sql_ops_space:
                    if query_tok == ')':
                        right_bracket += 1
                    if ((query_tok == '>' or query_tok == '<') and query_tok_list[idx+1] == '=') or (query_tok == ')' and (idx + 1 == len(query_tok_list) or query_tok_list[idx+1] == ',')):
                        # >= or <= ),
                        sub_query_format += query_tok
                        sub_query += query_tok
                    else:
                        sub_query_format += query_tok + ' '
                        sub_query += query_tok + ' '
                    last_tok = 'op'
                elif query_tok in sql_ops_no_space:
                    if query_tok == '(':
                        left_bracket += 1
                    sub_query_format += query_tok
                    sub_query += query_tok
                elif query_tok in column_names or '.' in query_tok:
                    if last_tok == 'val':
                        content_label += query_tok + ' '
                        continue
                    if idx + 1 == len(query_tok_list) or query_tok_list[idx+1] in [',', ')']:
                        sub_query_format += '[col]'
                        sub_query += query_tok
                    else:
                        sub_query_format += '[col] '
                        sub_query += query_tok + ' '
                    content_label += '[col] ' + query_tok + ' '
                    last_tok = 'col'
                elif query_tok in sql_marks:
                    sub_query_format += query_tok + ' '
                    sub_query += query_tok + ' '
                    last_tok = 'mark' 
                else:
                    if last_tok != 'val':
                        sub_query_format += '[val] '
                        content_label += '[val] '
                    if query_tok == '``':
                        sub_query += '"'
                        content_label += '"'
                    elif query_tok == "''":
                        sub_query += '" '
                        content_label += '" '
                    elif query_tok == "'":
                        sub_query += "' "
                        content_label += "' "
                    elif last_tok == 'val' and (idx + 1 == len(query_tok_list) or query_tok_list[idx+1] in ["'", '"', '``', "''"]):
                        sub_query += query_tok
                        content_label += query_tok
                    else:
                        sub_query += query_tok + ' '
                        content_label += query_tok + ' '
                    last_tok = 'val'

            if select_num > 1 and left_bracket > right_bracket:
                sub_query_format += ')'

            sub_query_format_list.append(sub_query_format)
            sub_query_list.append(sub_query)
            if data_training_args.stage == 'structure':
                structure = normalize(' '.join(sub_query_format_list))
                inputs.append(data_training_args.schema_serialization_with_prompt + ' | ' + input_str)
                target = geoquery_get_target(
                    query=' '.join(sub_query_format_list),
                    db_id="db",
                    normalize_query=True,
                    target_with_db_id=False,
                )
                targets.append(target)

            elif data_training_args.stage == 'content':

                random_number = random.random()
                random_idx = -1
                if mode == 'eval':
                    input_str = data_training_args.schema_serialization_with_prompt + eval_format_list[count] + ' | ' + input_str
                else:
                    input_str = data_training_args.schema_serialization_with_prompt + ' '.join(sub_query_format_list) + ' | ' + input_str
                inputs.append(input_str)
                target = content_label
                targets.append(target)
            count += 1

    else:
        inputs = [
            geoquery_get_input(question=question+"?", serialized_schema=serialized_schema, prefix=prefix)
            for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
        ]
        targets = [
            geoquery_get_target(
                query=sql,
                db_id="geo",
                normalize_query=data_training_args.normalize_query,
                target_with_db_id=data_training_args.target_with_db_id,
            )
            for sql in batch["query"]
        ]
    print(f"{mode}: {len(inputs)}")

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class GeoQueryTrainer(Seq2SeqTrainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        metas = [
            {
                "query": x["query"],
                "question": x["question"],
                "context": context,
                "label": label,
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        assert len(metas) == len(predictions)
        if self.stage == 'content':
            final_pred_sqls = []
            hypotheses_path = os.path.join(self.args.output_dir, "hypotheses.json")
            if os.path.exists(hypotheses_path):
                # sentence-level check
                with open(hypotheses_path) as f:
                    hypotheses = json.load(f)
                    for idx, item in enumerate(hypotheses):
                        db_id, structure = item["structure"].split(" | ")
                        db = os.path.join(metas[idx]["db_path"], db_id, f'{db_id}.sqlite')
                        schema = Schema(get_schema(db))
                        final_pred_sql = None
                        for hypothesis in item["topk_preds"]:
                            try:
                                pred_sql = combine_SC(content=hypothesis, input="", structure=structure)
                                parse_sql = get_sql(schema, pred_sql)
                                final_pred_sql = pred_sql
                                break                           
                            except:
                                continue
                        if final_pred_sql == None:
                            # default to the first one
                            final_pred_sql = combine_SC(content=item["topk_preds"][0], input="", structure=structure)
                        final_pred_sqls.append(final_pred_sql)
                            
                os.remove(hypotheses_path)
            else:
                for pred_content, meta in zip(predictions, metas):
                    final_pred_sqls.append(combine_SC(pred_content, meta['context'])) 
            # write predict sql
            with open(f"{self.args.output_dir}/predict_sql.txt", "w") as f:
                for final_pred_sql in final_pred_sqls:
                    f.write(final_pred_sql+"\n")

            with open(f"{self.args.output_dir}/content_{stage}.json", "w") as f:
                json.dump(
                    [dict(**{"input": meta['context']}, **{"prediction": prediction}, **{"label": label}, **{"score": prediction==label}, **{"pred_sql": final_pred_sql}, **{"gold_sql": meta['query']}) for meta, prediction, final_pred_sql, label in zip(metas, predictions, final_pred_sqls, decoded_label_ids)],
                    f,
                    indent=4,
                )
            return EvalPrediction(predictions=final_pred_sqls, label_ids=decoded_label_ids, metas=metas)
        elif self.stage == 'structure':        
            for idx in range(len(predictions)):
                if 'before' in predictions[idx]:
                    predictions[idx] = predictions[idx].replace('before', '<')
                if 'after' in predictions[idx]:
                    predictions[idx] = predictions[idx].replace('after', '>')
            return EvalPrediction(predictions=predictions, label_ids=decoded_label_ids, metas=metas)
        
    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        if self.stage == 'structure':
            accuracy = []
            accuracy.extend(
                (
                    pred.lower() == label.lower()
                    for pred, label in zip(predictions, label_ids)
                )
            )
            eval_metric = np.mean(accuracy)
            test_suite = dict()
            if eval_metric >= self.best_acc:
                with open(f"{self.args.output_dir}/structure.json", "w") as f:
                    json.dump(
                        [dict(**{"input": meta['context']}, **{"prediction": prediction}, **{"label": label}, **{"score": prediction==label}) for meta, prediction, label in zip(metas, predictions, label_ids)],
                        f,
                        indent=4,
                    )
            return {**{"exact_match": eval_metric}, **test_suite}
        elif self.stage == 'content':
            db_path = metas[0]["db_path"]
            db_id = "geo"
            tables = [dump_db_json_schema(os.path.join(db_path, db_id, f"{db_id}.sqlite"), db_id)]
            kmaps = build_foreign_key_map_from_json(tables)
            golds = []
            for x in metas:
                golds.append(x['query'])
            eval_metric = evaluate(golds, predictions, db_path, db_id, 'match', kmaps)["total_scores"]["all"]["exact"]
            test_suite = dict()
            return {**{"exact_match": eval_metric}, **test_suite}
        else:
            raise NotImplementedError()

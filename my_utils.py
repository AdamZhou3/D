
import os
import tqdm
import zipfile

import pandas as pd
import numpy as np
from shapely import wkt
import geopandas as gpd

from urllib.parse import urlparse
from requests import get
## Constanst =============================================
VERSION = "v3.0"  # output version

POICODE = {
"0101":"Residential|Gated Development",
"0102":"Residential|Private Development",
"0103":"Residential|Public Housing Development",
"0104":"Residential|Constituent",
"0105":"Residential|Other",
"0201":"Education|Public Elementary School",
"0202":"Education|Public Junior High-Intermediate-Middle",
"0203":"Education|Public High School",
"0204":"Education|Private/Parochial Elementary School",
"0205":"Education|Private/Parochial Junior/Middle School",
"0206":"Education|Private/Parochial High School",
"0207":"Education|Post Secondary Degree Granting Institution",
"0208":"Education|Other",
"0209":"Education|Public Early Childhood",
"0210":"Education|Public K-8",
"0211":"Education|Public K-12 all grades",
"0212":"Education|Public Secondary School",
"0213":"Education|Public School Building",
"0214":"Education|Public School Annex",
"0215":"Education|Private/Parochial Early Childhood",
"0216":"Education|Private/Parochial K-8",
"0217":"Education|Private/Parochial K-12 all grades",
"0218":"Education|Private/Parochial Secondary School",
"0301":"Cultural|Center",
"0302":"Cultural|Library",
"0303":"Cultural|Theater/Concert Hall",
"0304":"Cultural|Museum",
"0305":"Cultural|Other",
"0401":"Recreational|Park",
"0402":"Recreational|Amusement Park",
"0403":"Recreational|Golf Course",
"0404":"Recreational|Beach",
"0405":"Recreational|Botanical Garden",
"0406":"Recreational|Zoo",
"0407":"Recreational|Recreational Center",
"0408":"Recreational|Sports",
"0409":"Recreational|Playground",
"0410":"Recreational|Other",
"0411":"Recreational|Pool",
"0412":"Recreational|Garden",
"0501":"Social Services|Residential Child Care",
"0502":"Social Services|Day Care Center",
"0503":"Social Services|Adult Day Care",
"0504":"Social Services|Nursing Home/Assisted Living Facility",
"0505":"Social Services|Homeless shelter",
"0506":"Social Services|Other",
"0601":"Transportation|Bus Terminal",
"0602":"Transportation|Ferry landing/terminal",
"0603":"Transportation|Transit/Maintenance Yard",
"0604":"Transportation|Airport",
"0605":"Transportation|Heliport",
"0606":"Transportation|Marina",
"0607":"Transportation|Pier",
"0608":"Transportation|Bridge",
"0609":"Transportation|Tunnel",
"0610":"Transportation|Exit/Entrance",
"0611":"Transportation|Water Navigation",
"0612":"Transportation|Other",
"0701":"Commercial|Center",
"0702":"Commercial|Business",
"0703":"Commercial|Market",
"0704":"Commercial|Hotel/Motel",
"0705":"Commercial|Restaurant",
"0706":"Commercial|Other",
"0801":"Government|Government Office",
"0802":"Government|Court of law",
"0803":"Government|Post Office",
"0804":"Government|Consulate",
"0805":"Government|Embassy",
"0806":"Government|Military",
"0807":"Government|Other",
"0901":"Religious|Church",
"0902":"Religious|Synagogue",
"0903":"Religious|Temple",
"0904":"Religious|Convent/Monastery",
"0905":"Religious|Mosque",
"0906":"Religious|Other",
"1001":"Health|Hospital",
"1002":"Health|Inpatient care center",
"1003":"Health|Outpatient care center/Clinic",
"1004":"Health|Other",
"1101":"Public Safety|NYPD Precinct",
"1102":"Public Safety|NYPD Checkpoint",
"1103":"Public Safety|FDNY Ladder Company",
"1104":"Public Safety|FDNY Battalion",
"1105":"Public Safety|Correctional Facility",
"1106":"Public Safety|FDNY Engine Company",
"1107":"Public Safety|FDNY Special Unit",
"1108":"Public Safety|FDNY Division",
"1109":"Public Safety|FDNY Squad",
"1110":"Public Safety|NYPD Other",
"1111":"Public Safety|Other",
"1112":"Public Safety|FDNY Other",
"1201":"Water|Island",
"1202":"Water|River",
"1203":"Water|Lake",
"1204":"Water|Stream",
"1205":"Water|Other",
"1206":"Water|Pond",
"1301":"Miscellaneous|Official Landmark",
"1302":"Miscellaneous|Point of Interest",
"1303":"Miscellaneous|Cemetery/Morgue",
"1304":"Miscellaneous|Other"}

ZONINGCODE = {
'BPC': 'PU',
'C1-6': 'CR',
'C1-6A': 'CR',
'C1-7': 'CR',
'C1-7A': 'CR',
'C1-8': 'CR',
'C1-8A': 'CR',
'C1-8X': 'CR',
'C1-9': 'CR',
'C1-9A': 'CR',
'C2-6': 'CR',
'C2-6A': 'CR',
'C2-7': 'CR',
'C2-7A': 'CR',
'C2-8': 'CR',
'C2-8A': 'CR',
'C3': 'CR',
'C3A': 'CR',
'C4-1': 'CC',
'C4-2': 'CC',
'C4-2A': 'CC',
'C4-2F': 'CC',
'C4-3': 'CC',
'C4-3A': 'CC',
'C4-4': 'CC',
'C4-4A': 'CC',
'C4-4D': 'CC',
'C4-4L': 'CC',
'C4-5': 'CC',
'C4-5A': 'CC',
'C4-5D': 'CC',
'C4-5X': 'CC',
'C4-6': 'CC',
'C4-6A': 'CC',
'C4-7': 'CC',
'C5-1': 'CC',
'C5-1A': 'CC',
'C5-2': 'CC',
'C5-2.5': 'CC',
'C5-2A': 'CC',
'C5-3': 'CC',
'C5-4': 'CC',
'C5-5': 'CC',
'C5-P': 'CC',
'C6-1': 'CC',
'C6-1A': 'CC',
'C6-1G': 'CC',
'C6-2': 'CC',
'C6-2A': 'CC',
'C6-2G': 'CC',
'C6-2M': 'CC',
'C6-3': 'CC',
'C6-3A': 'CC',
'C6-3D': 'CC',
'C6-3X': 'CC',
'C6-4': 'CC',
'C6-4.5': 'CC',
'C6-4A': 'CC',
'C6-4M': 'CC',
'C6-4X': 'CC',
'C6-5': 'CC',
'C6-5.5': 'CC',
'C6-6': 'CC',
'C6-6.5': 'CC',
'C6-7': 'CC',
'C6-7T': 'CC',
'C6-9': 'CC',
'C7': 'CR',
'C8-1': 'CM',
'C8-2': 'CM',
'C8-3': 'CM',
'C8-4': 'CM',
'M1-1': 'MU',
'M1-1/R5': 'MU',
'M1-1/R7-2': 'MU',
'M1-1/R7D': 'MU',
'M1-1D': 'MU',
'M1-2': 'MU',
'M1-2/R5B': 'MU',
'M1-2/R5D': 'MU',
'M1-2/R6': 'MU',
'M1-2/R6A': 'MU',
'M1-2/R6B': 'MU',
'M1-2/R7A': 'MU',
'M1-2/R8': 'MU',
'M1-2/R8A': 'MU',
'M1-2D': 'MU',
'M1-3': 'MU',
'M1-3/R7X': 'MU',
'M1-3/R8': 'MU',
'M1-4': 'MU',
'M1-4/R6A': 'MU',
'M1-4/R6B': 'MU',
'M1-4/R7-2': 'MU',
'M1-4/R7A': 'MU',
'M1-4/R7X': 'MU',
'M1-4/R8A': 'MU',
'M1-4D': 'MU',
'M1-5': 'MU',
'M1-5/R10': 'MU',
'M1-5/R7-2': 'MU',
'M1-5/R7-3': 'MU',
'M1-5/R7X': 'MU',
'M1-5/R8A': 'MU',
'M1-5/R9': 'MU',
'M1-5/R9-1': 'MU',
'M1-5A': 'MU',
'M1-5B': 'MU',
'M1-5M': 'MU',
'M1-6': 'MU',
'M1-6/R10': 'MU',
'M1-6D': 'MU',
'M2-1': 'MU',
'M2-2': 'MU',
'M2-3': 'MU',
'M2-4': 'MU',
'M3-1': 'MU',
'M3-2': 'MU',
'PARK': 'PU',
'R1-1': 'RL',
'R1-2': 'RL',
'R1-2A': 'RL',
'R10': 'RH',
'R10A': 'RH',
'R10H': 'RH',
'R2': 'RL',
'R2A': 'RL',
'R2X': 'RL',
'R3-1': 'RL',
'R3-2': 'RL',
'R3A': 'RL',
'R3X': 'RL',
'R4': 'RL',
'R4-1': 'RL',
'R4A': 'RL',
'R4B': 'RL',
'R5': 'RL',
'R5A': 'RL',
'R5B': 'RL',
'R5D': 'RL',
'R6': 'RH',
'R6A': 'RH',
'R6B': 'RH',
'R7-1': 'RH',
'R7-2': 'RH',
'R7-3': 'RH',
'R7A': 'RH',
'R7B': 'RH',
'R7D': 'RH',
'R7X': 'RH',
'R8': 'RH',
'R8A': 'RH',
'R8B': 'RH',
'R8X': 'RH',
'R9': 'RH',
'R9A': 'RH',
'R9X': 'RH'}
## Functions ============================================
def cache_data(src:str, dest:str, fn =None )->str:
    '''
    args:
        src: source url
        dest: destination directory
    return:
        dfn: destination directory + filename
    '''
    url = urlparse(src) # We assume that this is some kind of valid URL
    if fn is None:
        fn = os.path.split(url.path)[-1] # Extract the filename
    dfn = os.path.join(dest,fn) # Destination filename
    
    if not os.path.isfile(dfn):
        
        print(f"{dfn} not found, downloading!")
        path = os.path.split(dest)
        
        if len(path) >= 1 and path[0] != '':
            os.makedirs(os.path.join(*path), exist_ok=True)
            
        with open(dfn, "wb") as file:
            response = get(src)
            file.write(response.content)
            
        print(f"\tDone downloading.")

    else:
        print(f"Found {dfn} locally!")
        
    return dfn

def unzip_file(mzip, mdir):
    """
    Extract files from zip without keeping the structure
    args:
        mzip: dir + filename
        mdir: destiny dir
    """
    path = os.path.split(mdir)

    if len(path) >= 1 and path[0] != '':
        os.makedirs(os.path.join(*path), exist_ok=True)
    
    with zipfile.ZipFile(mzip) as zipf:
        for zip_info in zipf.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zipf.extract(zip_info, mdir)
    print(f"unziped {mzip}")
    
def date_features(df, date="date"):
    """
    Extracte date features based on the timestamp column in the data frame
    """
    df[date] = pd.to_datetime(df[date])
#     df[date+"_month"] = df[date].dt.month.astype(int)
#     df[date+"_year"]  = df[date].dt.year.astype(int)
#     df[date+"_week"]  = df[date].dt.week.astype(int)
    df[date+"_day"]   = df[date].dt.day.astype(int)
#     df[date+"_dayofweek"]= df[date].dt.dayofweek.astype(int)
#     df[date+"_dayofyear"]= df[date].dt.dayofyear.astype(int)
    df[date+"_hour"] = df[date].dt.hour.astype(int)
#     df[date+"_int"] = pd.to_datetime(df[date]).astype(int)
    df[date+"_weekend"] = np.where(df[date].dt.dayofweek < 5,0,1)
    return df

def get_sequences_by_distancegreedy(gdf_tz, gdf_poi):
    sequences={}
    gdf_tz["geometry"] =  gdf_tz.apply(lambda x:Polygon(x["geometry"].buffer(100).exterior), axis=1)
    df_join = sjoin(gdf_poi, gdf_tz, how="inner",op="within")
    
    for tz_ind in tqdm.tqdm(df_join.index_right.unique()):
        tz_pois = df_join[df_join.index_right==tz_ind].reset_index()

        if tz_pois.shape[0]>0:
            ## create distance matrix for each two points
            pnt_num = tz_pois.shape[0]
            dismat = np.zeros(shape=(pnt_num,pnt_num),dtype=int)
            for i in range(pnt_num):
                for j in range(pnt_num):
                    dismat[i,j] = int(tz_pois.loc[i,"geometry"].distance(tz_pois.loc[j,"geometry"]))

            ## get the pair with largest distance
            visited = list(np.unravel_index(np.argmax(dismat, axis=None), dismat.shape))
            
            ## list of to be visited points
            not_visited = [x for x in range(pnt_num) if x not in visited]

            np.random.shuffle(not_visited)

            while not_visited:
                to_be_visit = not_visited.pop()
                if len(visited)==2:
                    visited.insert(1,to_be_visit)
                    pass
                else:
                    ## find the index to insert
                    search_bound = list(zip(visited[0:-1],visited[1:]))
                    dis = [dismat[to_be_visit,x]+dismat[to_be_visit,y] for x,y in search_bound]
                    insert_place = dis.index(min(dis))+1

                    visited.insert(insert_place,to_be_visit)
            sequences[tz_ind] = tz_pois.loc[visited,"code"].values
            # yield sequences
    return sequences

def read_csv_to_gdf(path, col="geometry", crs="epsg:4326"):
    """
    args:
        path: path to csv file 
        col: geopandas geometry column
        crs: projection
    return:
        gdf: geodataframe
    """
    df = pd.read_csv(path)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry",crs=crs)
    return gdf

if __name__ == '__main__':
    main()
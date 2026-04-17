import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
import openpyxl

arq_parquet = pq.ParquetFile('fhvhv_tripdata_2020-01.parquet')
tabela = pq.read_table('fhvhv_tripdata_2020-01.parquet')
df = tabela.to_pandas()

print(df.shape)
print(df.dtypes)

df.isna().sum()
df.loc[df["trip_miles"].idxmax(), ["trip_miles", "trip_time", "PULocationID", "DOLocationID"]]

for col in df.select_dtypes(include=['object', 'string']).columns:
    if df[col].str.startswith(" ").any() or df[col].str.endswith(" ").any():    
        df[col] = df[col].str.strip()

taxi_zone = pd.read_excel('TAXI_ZONE.xlsx')
mapa_cidade = dict(zip(taxi_zone['LocationID'], taxi_zone['Borough']))

total_linhas = 0
primeiro_chunk = True
writer = None
colunas_finais = None

for batch in arq_parquet.iter_batches(batch_size=15000):
    chunk = batch.to_pandas()

    chunk = chunk.drop(columns=[
        'originating_base_num', 'on_scene_datetime', 'sales_tax', 
        'congestion_surcharge', 'airport_fee'
    ], errors='ignore')

    chunk = chunk.drop_duplicates()

    chunk['driver_pay'] = (chunk['driver_pay'] - chunk['tolls']) + chunk['bcf'] + chunk['tips']
    chunk['distance_traveled'] = (chunk['trip_miles'] * 1.609).round(2)
    chunk['trip_time'] = (chunk['trip_time'] / 60).round(2)

    chunk = chunk[chunk['driver_pay'] != 0]

    chunk['PU_Borough'] = chunk['PULocationID'].map(mapa_cidade)
    chunk['DO_Borough'] = chunk['DOLocationID'].map(mapa_cidade)

    chunk['hvfhs_license_num'] = chunk['hvfhs_license_num'].replace({
        'HV0002': 'Juno',
        'HV0003': 'Uber',
        'HV0004': 'Via',
        'HV0005': 'Lyft'
    })

    chunk = chunk.dropna(subset=['PU_Borough'])
    
    colunas_flags = ['shared_request_flag', 'shared_match_flag', 'access_a_ride_flag', 
                     'wav_request_flag', 'wav_match_flag']
    
    for coluna in colunas_flags:
        if coluna in chunk.columns:
            chunk[coluna] = chunk[coluna].fillna('N')
            chunk[coluna] = chunk[coluna].replace('', 'N')
            chunk[coluna] = chunk[coluna].replace(' ', 'N')
    
    vazios_do = chunk['DO_Borough'].isna().sum()
    if vazios_do > 0:
        chunk['DO_Borough'] = chunk['DO_Borough'].fillna('Outside of NYC')

    chunk = chunk.drop(columns=['tolls', 'bcf', 'tips', 'trip_miles'])

    chunk = chunk.reset_index(drop=True)

    if primeiro_chunk:
        colunas_finais = chunk.columns.tolist()
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        writer = pq.ParquetWriter('fhvhv_tripdata_clean_v2.parquet', table.schema, compression='snappy')
        writer.write_table(table)
        primeiro_chunk = False
    else:
        chunk = chunk[colunas_finais]
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        
        if not writer.is_open:
            writer = pq.ParquetWriter('fhvhv_tripdata_clean_v2.parquet', table.schema, compression='snappy')
        
        writer.write_table(table)

if writer and writer.is_open:
    writer.close()

tabela_limpa = pq.read_table('fhvhv_tripdata_clean_v2.parquet')
df_limpo = tabela_limpa.to_pandas()

print(df_limpo.shape)
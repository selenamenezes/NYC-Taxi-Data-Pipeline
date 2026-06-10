# NYC TAXI DATA PIPELINE — Limpeza & Enriquecimento em Python

Este repositório/pipeline contém um script em Python para **processar arquivos Parquet** de dados de viagens (ex.: NYC Taxi Green/Yellow/High Volume for Hire), realizar **limpeza**, **transformações** e **enriquecimento com dados auxiliares** (taxi zone), gerando ao final um **Parquet limpo** pronto para análise.

> O script foi escrito para processar o arquivo `fhvhv_tripdata_2020-01.parquet` em **chunks (batch)**, evitando estourar memória em datasets grandes.

## Funcionalidades principais

### 1) Leitura por Parquet
- Abre o Parquet com `pyarrow.parquet.ParquetFile`.
- Faz leitura em lotes via `iter_batches(batch_size=15000)`.

### 2) Limpeza e padronização
- Remove colunas que não serão usadas na etapa final (ex.: `originating_base_num`, `on_scene_datetime`, taxas/fees específicas), usando `errors='ignore'` para tolerar variações.
- Remove duplicados (`drop_duplicates`).
- Remove espaços extras em colunas `object/string` com `strip()` quando necessário.

### 3) Transformações de métricas
- Calcula `driver_pay` somando/ajustando valores com base em:
  - `(driver_pay - tolls) + bcf + tips`
- Cria/transforma campos:
  - `distance_traveled` em km (converte de milhas para km)
  - `trip_time` em minutos (converte de segundos para minutos, conforme divisão por 60)
- Remove registros com `driver_pay == 0`.
- (Opcional) Atribui ou prepara campos antes do enriquecimento.

### 4) Enriquecimento por mapeamento de bairro (PU/DO)
- Carrega `TAXI_ZONE.xlsx`.
- Cria um dicionário `LocationID -> Borough`.
- Mapeia:
  - `PU_Borough` a partir de `PULocationID`
  - `DO_Borough` a partir de `DOLocationID`
- Remove linhas onde `PU_Borough` é nulo.
- Para `DO_Borough` nulo, preenche com `Outside of NYC`.

### 5) Normalização de licença HVFHS
- Substitui códigos de `hvfhs_license_num` para nomes:
  - `HV0002 -> Juno`, `HV0003 -> Uber`, `HV0004 -> Via`, `HV0005 -> Lyft`.

### 6) Tratamento de flags
- Para colunas booleanas/flags específicas (ex.: `shared_request_flag`, `shared_match_flag`, etc.):
  - preenche `NaN` com `'N'`
  - normaliza valores vazios e strings com espaços para `'N'`

### 7) Saída final: Parquet limpo
- Mantém o schema das colunas do primeiro chunk.
- Escreve incrementalmente em `fhvhv_tripdata_clean_v2.parquet` com:
  - `ParquetWriter`
  - compressão `snappy`
- Ao final, lê novamente o Parquet limpo e converte para pandas para inspeção (`shape`, dtypes, etc.).

## Arquivos esperados

- `fhvhv_tripdata_2020-01.parquet` (entrada)
- `TAXI_ZONE.xlsx` (mapeamento `LocationID -> Borough`)

## Saída gerada

- `fhvhv_tripdata_clean_v2.parquet`

## Dependências

- `pyarrow`
- `pandas`
- `openpyxl`

Instalação (exemplo):
```bash
pip install pandas pyarrow openpyxl
```

## Como executar

1. Garanta que os arquivos de entrada estejam na mesma pasta do script.
2. Execute:
```bash
python script.py
```

## Notas e cuidados
- A lógica depende de nomes de colunas específicos (ex.: `trip_miles`, `trip_time`, `PULocationID`, `DOLocationID`, `hvfhs_license_num`, flags de compartilhamento).
- O script tenta ser tolerante para algumas colunas usando `errors='ignore'`, mas o restante assume a presença das colunas usadas em transformações e filtros.
- Para datasets muito grandes, ajuste `batch_size`.

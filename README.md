# ContextSeqRecETL

ContextSeqRecETL prepara o dataset de imóveis consumido pelo `ContextSeqRec`. Ele lê listings e eventos regionais em CSV compactado, usa Spark para gerar Parquets consolidados e produz o `dataset.pkl` usado no treinamento.

O fluxo é independente de outros repositórios:

```text
CSV.gz brutos
    -> contextseqrec-etl convert
    -> events.parquet + items.parquet
    -> contextseqrec-etl preprocess
    -> ContextSeqRec/data/dataset.pkl
```

Dados brutos, Parquets, temporários Spark e `dataset.pkl` devem ficar fora deste diretório e não devem ser versionados.

## Pré-requisitos

Use Linux com:

- Python 3.12;
- Java 11;
- espaço em disco para os CSVs, intermediários Spark e Parquets finais;
- memória suficiente para o driver Spark e para a conversão pandas final.

A preparação não precisa de GPU.

Confirme o ambiente:

```bash
python3.12 --version
java -version
```

## Instalação

No diretório do projeto:

```bash
cd /home/hygo2025/Development/projects/ContextSeqRecETL
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/contextseqrec-etl --help
```

`requirements.lock` fixa Spark, pandas, PyArrow, H3 e todas as dependências transitivas. `--require-hashes` verifica os arquivos baixados, e a segunda instalação apenas registra o código local sem resolver dependências novamente.

## Estrutura dos dados brutos

Escolha diretórios externos ao projeto. Por exemplo:

```bash
export RAW_ROOT=/mnt/dados/contextseqrec/raw
export ETL_WORK=/mnt/dados/contextseqrec/work
export SPARK_TMP=/mnt/dados/contextseqrec/spark-tmp
export CONTEXTSEQREC_ROOT=/home/hygo2025/Development/projects/ContextSeqRec
```

Os CSVs devem estar organizados por tipo e região:

```text
$RAW_ROOT/
├── listings/
│   ├── mg/*.csv.gz
│   └── es_bh/*.csv.gz
└── events/
    ├── mg/*.csv.gz
    └── es_bh/*.csv.gz
```

Os nomes singulares `listing/` e `event/` também são aceitos. Os arquivos precisam ser CSVs com cabeçalho e extensão `.csv.gz`.

### Colunas obrigatórias de listings

```text
anonymized_listing_id
status
updated_at
lat_region
lon_region
business_type
usage_type
```

Somente listings com `status=ACTIVE` permanecem. Quando há mais de uma linha para o mesmo `anonymized_listing_id`, o ETL mantém a linha de `updated_at` mais recente.

As colunas abaixo são opcionais e preservadas quando existem:

```text
price
usable_areas
total_areas
bedrooms
bathrooms
suites
parking_spaces
zip_code
neighborhood
unit_type
amenities
```

### Colunas obrigatórias de events

```text
anonymized_listing_id
anonymized_session_id
collector_timestamp
event_type
```

`collector_timestamp` deve conter milissegundos desde o Unix epoch. Os comportamentos aceitos são:

```text
RankingClicked
ListingRendered
FavoriteClicked
GalleryClicked
ShareClicked
DecisionTreeFormClicked
LeadPanelClicked
LeadClicked
```

Outros comportamentos são descartados. Eventos ligados a listings que não permaneceram ativos também são removidos.

## Etapa 1: converter CSV.gz em Parquet

Crie os diretórios externos:

```bash
mkdir -p "$ETL_WORK" "$SPARK_TMP"
```

Configure os recursos do Spark. Estes valores são operacionais e devem caber na máquina:

```bash
export SPARK_LOCAL_DIR="$SPARK_TMP"
export SPARK_CORES=16
export SPARK_DRIVER_MEMORY=100g
export SPARK_SHUFFLE_PARTITIONS=400
```

Em uma máquina menor, por exemplo, use:

```bash
export SPARK_CORES=8
export SPARK_DRIVER_MEMORY=24g
export SPARK_SHUFFLE_PARTITIONS=200
```

Execute a conversão das regiões `mg` e `es_bh`:

```bash
cd /home/hygo2025/Development/projects/ContextSeqRecETL
.venv/bin/contextseqrec-etl convert \
  --data-root "$RAW_ROOT" \
  --out-dir "$ETL_WORK" \
  --regions mg es_bh \
  --min-session-length 3 \
  --max-session-length 50 \
  --min-item-freq 5 \
  --h3-res 6 7 \
  --output-partitions 200
```

A conversão:

1. seleciona a versão ativa mais recente de cada listing;
2. calcula células H3 nas resoluções informadas;
3. junta os eventos ao catálogo ativo;
4. ordena os eventos por sessão;
5. colapsa rajadas consecutivas do mesmo item e comportamento;
6. usa a quantidade de eventos da rajada como intensidade;
7. filtra sessões curtas, trunca sessões longas e remove itens raros;
8. cria IDs densos positivos;
9. consolida as regiões em Parquets finais.

O resultado esperado é:

```text
$ETL_WORK/
├── mg/
│   ├── listings_processed/
│   ├── listing_id_mapping/
│   ├── events_processed/
│   ├── sessions_processed/
│   └── items.parquet/
├── es_bh/
│   └── ...
├── events.parquet/
└── items.parquet/
```

### Retomar a conversão

Cada diretório Spark concluído contém `_SUCCESS`. Repetir o mesmo comando ignora etapas regionais completas e refaz o merge final.

Se todos os outputs regionais já existem e apenas o merge precisa ser refeito:

```bash
.venv/bin/contextseqrec-etl convert \
  --data-root "$RAW_ROOT" \
  --out-dir "$ETL_WORK" \
  --regions mg es_bh \
  --output-partitions 200 \
  --final-only
```

Se regiões, filtros ou dados de entrada mudarem, use um novo `ETL_WORK`. Não reutilize intermediários produzidos com parâmetros científicos diferentes.

## Etapa 2: gerar dataset.pkl

O preprocessamento final lê `events.parquet` e `items.parquet`, deriva o segmento de mercado, ordena cada sessão, densifica IDs e separa o último `LeadClicked` elegível.

Gere diretamente o arquivo esperado pelo `ContextSeqRec`:

```bash
mkdir -p "$CONTEXTSEQREC_ROOT/data"
cd /home/hygo2025/Development/projects/ContextSeqRecETL
.venv/bin/contextseqrec-etl preprocess \
  --source-dir "$ETL_WORK" \
  --output "$CONTEXTSEQREC_ROOT/data/dataset.pkl" \
  --min-user-events 5 \
  --target-behavior LeadClicked
```

O comando recusa sobrescrever um arquivo existente. Para substituir conscientemente o arquivo:

```bash
.venv/bin/contextseqrec-etl preprocess \
  --source-dir "$ETL_WORK" \
  --output "$CONTEXTSEQREC_ROOT/data/dataset.pkl" \
  --min-user-events 5 \
  --target-behavior LeadClicked \
  --force
```

A publicação é atômica: o pickle é escrito em um temporário no diretório de destino e renomeado somente depois de concluído.

`dataset.pkl` usa pickle. Abra apenas arquivos produzidos por este ETL ou por uma fonte confiável.

## Etapa 3 (opcional): exportar features de item

Gera uma matriz densa de atributos de item `[num_items + 1, F]` alinhada ao
`smap` do `dataset.pkl`, consumida opcionalmente pelo `ContextSeqRec` em
`sampling.item_features_path`. É pandas puro; não usa Spark.

A linha `0` é o item de padding e é sempre zero. A linha `i` (1..M) contém os
atributos do item denso `i` usado pelo modelo. O alinhamento é feito assim:

```text
events.parquet: sid -> canonical_listing_id
items.parquet : canonical_listing_id -> atributos
dataset.pkl   : smap[sid] = id denso
```

Nunca se usa `listing_id_numeric` (regional, sem offset). As transformações são
fixas, sem estatística ajustada: `log1p` em preço/áreas, indicadores de missing,
one-hot de categóricos, multi-hot de amenities e coordenadas com indicador de
ausência. Isso mantém a exportação determinística e sem vazamento de
validação/teste (nada depende do split, de contagens de interação ou de rótulos).

```bash
cd /home/hygo2025/Development/projects/ContextSeqRecETL
.venv/bin/contextseqrec-etl export-item-features \
  --dataset "$CONTEXTSEQREC_ROOT/data/dataset.pkl" \
  --source-dir "$ETL_WORK" \
  --output "$CONTEXTSEQREC_ROOT/data/item_features.npy" \
  --amenities-top 64
```

O comando recusa sobrescrever a saída; use `--force` para substituir. Além do
`.npy`, grava um `item_features.npy.manifest.json` com hashes das entradas,
número de itens, dimensões e a lista de colunas. Guarde o `.npy`, o manifesto e o
`dataset.pkl` como um par inseparável e versionado.

## Conferir o resultado

Registre o SHA-256 do arquivo:

```bash
sha256sum "$CONTEXTSEQREC_ROOT/data/dataset.pkl"
```

Depois valide o arquivo com o próprio projeto de treino:

```bash
cd "$CONTEXTSEQREC_ROOT"
CUDA_VISIBLE_DEVICES="" .venv/bin/contextseqrec plan \
  --config configs/real_estate.toml
```

O comando deve preparar o dataset, criar o split determinístico e imprimir o plano sem iniciar treinamento.

Para executar a campanha após o preflight:

```bash
.venv/bin/contextseqrec run --config configs/real_estate.toml
```

O procedimento completo da tese, incluindo instalação do treino, várias GPUs, máquinas separadas, merge e conferência dos 40 runs, está em:

```text
../ContextSeqRec/THESIS_RUNBOOK.md
```

## Referência da CLI

Ajuda geral:

```bash
.venv/bin/contextseqrec-etl --help
```

Opções da conversão:

```bash
.venv/bin/contextseqrec-etl convert --help
```

Opções do preprocessamento:

```bash
.venv/bin/contextseqrec-etl preprocess --help
```

Opções da exportação de features de item:

```bash
.venv/bin/contextseqrec-etl export-item-features --help
```

## Problemas comuns

### `java: command not found`

Instale Java 11 e confirme `java -version` antes de iniciar o Spark.

### Nenhum `.csv.gz` encontrado

Confirme o layout, os nomes das regiões e a extensão dos arquivos. O ETL procura exatamente `*.csv.gz` dentro de cada região.

### Falta de memória ou espaço no Spark

Reduza `SPARK_CORES`, ajuste `SPARK_DRIVER_MEMORY`, aumente o espaço em `SPARK_LOCAL_DIR` e confira se o diretório temporário está em um disco adequado.

### `dataset.pkl` já existe

Escolha outro caminho de saída ou use `--force` somente quando a substituição for intencional.

### Mudança de parâmetros após uma execução parcial

Use outro diretório em `--out-dir`. A retomada por `_SUCCESS` pressupõe que dados, regiões e parâmetros são os mesmos.

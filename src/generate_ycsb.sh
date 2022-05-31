#!/usr/bin/env bash

set -e

#BASE_DIR="/hpi/fs00/home/lawrence.benson/clion/hykv1/benchmark"
BASE_DIR="/home/jiawen/dbms/hykv/benchmark"
PREFILL_CONF="${BASE_DIR}/config/ycsb_prefill.conf"
DATA_DIR="/home/jiawen/dbms/tools/ycsb_data"

#CONFIGS=( "5050_uniform" "5050_zipf" "1090_uniform" "1090_zipf" "9010_zipf")
#CONFIGS=( "9010_uniform" )
CONFIGS=( "9010_zipf" )

#cd "/hpi/fs00/home/lawrence.benson/ycsb"
cd "/home/jiawen/dbms/tools/ycsb"

echo "GENERATING PREFILL DATA"
./bin/ycsb load basic -P ${PREFILL_CONF} -s > "${DATA_DIR}/raw_prefill.dat"

echo "GENERATING YCSB DATA"
for config in "${CONFIGS[@]}"
do
  echo "GENERATING ${config}..."
  ./bin/ycsb run basic -P ${PREFILL_CONF} \
          -P "${BASE_DIR}/config/ycsb_${config}.conf" \
          -s > "${DATA_DIR}/raw_ycsb_wl_${config}.dat"
done


cd "${BASE_DIR}"
echo "CONVERTING DATA TO BINARY FORMAT"

python3 convert_ycsb.py "${DATA_DIR}/raw_prefill.dat" "${DATA_DIR}/ycsb_prefill.dat"

for config in "${CONFIGS[@]}"
do
  echo "CONVERTING: ${config}..."
  python3 convert_ycsb.py "${DATA_DIR}/raw_ycsb_wl_${config}.dat" "${DATA_DIR}/ycsb_wl_${config}.dat"
  rm "${DATA_DIR}/raw_ycsb_wl_${config}.dat"
done

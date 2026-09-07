mkdir -p data
cd data

gsutil -m cp \
  "gs://snapedit-vuhoang/vu_data/super-erase_object-effect/CausRem_extract.zip" \
  "gs://snapedit-vuhoang/vu_data/super-erase_object-effect/ObjectClear.zip" \
  "gs://snapedit-vuhoang/vu_data/super-erase_object-effect/data_good_qwen-251210.zip" \
  "gs://snapedit-vuhoang/vu_data/super-erase_object-effect/RORD_subset10k.zip" \
  .

unzip CausRem_extract.zip 
unzip ObjectClear.zip
unzip RORD_subset10k.zip
unzip data_good_qwen-251210.zip

rm *.zip
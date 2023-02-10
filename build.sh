python trojan_detector.py  --model_filepath=../trojai-datasets/round11-train-dataset/models/id-00000003/model.pt  --result_filepath=./output.txt  --scratch_dirpath=./scratch/  --examples_dirpath=../trojai-datasets/round11-train-dataset/models/id-00000003/clean-example-data  --round_training_dataset_dirpath=/path/to/training/dataset/ --round_training_dataset_dirpath ../trojai-datasets/coco2017/trojai_coco/  --metaparameters_filepath=./metaparameters.json  --schema_filepath=./metaparameters_schema.json  --learned_parameters_dirpath=./learned_parameters 


sudo singularity build image-classification-sep2022_sts_SRI_trinity_v0.simg trojan_detector.def 


singularity run --nv -B ../trojai-datasets:/data/ image-classification-sep2022_sts_SRI_trinity_v0.simg  --model_filepath=/data/round11-train-dataset/models/id-00000001/model.pt  --result_filepath=./output.txt  --scratch_dirpath=./scratch/  --examples_dirpath=/data/round11-train-dataset/models/id-00000001/clean-example-data --round_training_dataset_dirpath /data/coco2017/trojai_coco/  --round_training_dataset_dirpath=/path/to/training/dataset/  --metaparameters_filepath=/metaparameters.json  --schema_filepath=/metaparameters_schema.json  --learned_parameters_dirpath=/learned_parameters --example_img_format jpg

python trojan_detector.py --configure_mode  --configure_models_dirpath ../trojai-datasets/round11-train-dataset/models/  --learned_parameters_dirpath=./learned_parameters 

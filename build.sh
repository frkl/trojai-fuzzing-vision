
python entrypoint.py infer \
--model_filepath ../trojai-datasets/object-detection-feb2023v2/models_tmp/id-00000127/model.pt \
--result_filepath ./scratch/output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath ../trojai-datasets/object-detection-feb2023v2/models_tmp/id-00000127/clean-example-data \
--round_training_dataset_dirpath /path/to/train-dataset \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json 


sudo singularity build object-detection-feb2023_sts_SRI_trinity_v3.simg trojan_detector.def 


singularity run \
--bind /work2/project/trojai-datasets/object-detection-feb2023v2 \
--nv \
./object-detection-feb2023_sts_SRI_trinity_v3.simg \
infer \
--model_filepath /work2/project/trojai-datasets/object-detection-feb2023v2/models_tmp/id-00000127/model.pt \
--result_filepath=/output.txt \
--scratch_dirpath=/scratch/ \
--examples_dirpath /work2/project/trojai-datasets/object-detection-feb2023v2/models_tmp/id-00000127/clean-example-data \
--round_training_dataset_dirpath=/path/to/training/dataset/ \
--metaparameters_filepath=/metaparameters.json \
--schema_filepath=/metaparameters_schema.json \
--learned_parameters_dirpath=/learned_parameters/ 



#python entrypoint.py configure \
--scratch_dirpath ./scratch/ \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json \
--learned_parameters_dirpath ./learned_parameters \
--configure_models_dirpath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/models \
--scale_parameters_filepath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/scale_params.npy


#python entrypoint.py configure \
--scratch_dirpath ./scratch/ \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json \
--learned_parameters_dirpath ./learned_parameters \
--configure_models_dirpath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/models \
--scale_parameters_filepath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/scale_params.npy \
--automatic_configuration



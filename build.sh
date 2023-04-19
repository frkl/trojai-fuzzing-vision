
#python entrypoint.py infer \
--model_filepath ../trojai-datasets/object-detection-feb2023/models/id-00000002/model.pt \
--result_filepath ./scratch/output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath ../trojai-datasets/object-detection-feb2023/models/id-00000002/clean-example-data \
--round_training_dataset_dirpath /path/to/train-dataset \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json 


sudo singularity build object-detection-feb2023_sts_SRI_trinity_v1.simg trojan_detector.def 


singularity run \
--bind /work2/project/trojai-datasets/object-detection-feb2023 \
--nv \
./object-detection-feb2023_sts_SRI_trinity_v1.simg \
infer \
--model_filepath /work2/project/trojai-datasets/object-detection-feb2023/models/id-00000002/model.pt \
--result_filepath=/output.txt \
--scratch_dirpath=/scratch/ \
--examples_dirpath /work2/project/trojai-datasets/object-detection-feb2023/models/id-00000002/clean-example-data \
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



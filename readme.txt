this folder consists of:
1. Python file:
    1. <deploy_endpoint.ipynb>
    2. <model/code/inference.py> 




2. Pytorch model must be resaved using:
    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)



3. Model and inference must be save in the following structure:
    folder structure:
        model |
            model.pth
            code |
                inference.py

4. Model folder must be compressed to model.tar.gz 



5. Important:
    endpoint and ecs must be in the same vpc 
    vpc must have subnet and security group of the ecs
    user credentials used must have access to s3 and sagemake resources  











* functions in "model/code/inference.py" "script:
    a func. to install any needed library it will be used to install "s3fs"                      <install>
    a func. to read then load saved model in the model dir                                       <model_fn>
    a func. to load image from the given image s3 path in json forma                             <input_fn>
    a func. to input the loaded image to the model and get output                                <predict_fn>
    a func. to get the prediction from the output and return it in json format                   <output_fn>





Usage:
    Create sagemaker notebook instance and upload  <deploy_endpoint.ipynb>
    Edit the model path variable in the endpoint to be the path of the model.tar.gz on the s3 bucket
    Edit the endpoint_name with the prefered name
    Open the notebook then run
    








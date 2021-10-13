# HyperKvasir Tutorial


### 1. About dataset
The data is collected during real gastro- and colonoscopy examinations at a Hospital in Norway and partly labeled by experienced gastrointestinal endoscopists. For more information, please visit [this](https://datasets.simula.no/hyper-kvasir/) web site.


### 2. How to run this tutorial
* Run without TLS connection:

    1\. Run director:
    ```sh
    cd director
    fx director start --disable-tls -c director_config.yaml
    ```

    2\. Start first envoy:
    ```sh
    cd envoy
    pip install -r requirements.txt
    fx envoy start -n env_one --disable-tls --shard-config-path shard_config_one.yaml -dh localhost -dp 50051
    ```

    3\. Start second envoy:
    - Copy `envoy` folder to another place and run from there:
    ```sh
    pip install -r requirements.txt
    fx envoy start -n env_two --disable-tls --shard-config-path shard_config_two.yaml -dh localhost -dp 50051
    ```

    4\. Run `PyTorch_Kvasir_UNet.ipynb` jupyter notebook:
    ```sh
    cd workspace
    jupyter notebook PyTorch_Kvasir_UNet.ipynb
    ```

* Run with TLS connection:

    1\. Create CA:
    ```sh
    fx pki install -p ./CA
    ```

    ---
    **_NOTE:_** By default, server will run on `localhost:9123`, if you want to change this, please run:
    ```sh
    fx pki install -p ./CA --ca-url <host:port>
    ```
    ---

    2\. Run CA https server:
    ```sh
    fx pki run -p ./CA
    ```

    3\. Create neccesary tokens:

    * Create token for director:

    ```sh
    fx pki get-token -n localhost --ca-path ./CA
    ```


    * Create token for first envoy:

    ```sh
    fx pki get-token -n env_one --ca-path ./CA
    ```

    * Create token for second envoy:

    ```sh
    fx pki get-token -n env_two --ca-path ./CA
    ```
    ---
    **_NOTE:_**  If your CA server is not running on the `localhost:9123`, then please specify the host and port:
    ```sh
    fx pki get-token -n <name> --ca-path ./CA --ca-url <host:port>
    ```
    ---

    * Save all tokens to a file or don't close the terminal
    
    4\. Certify all nodes:

    On this step, you have to copy token to node side (director or envoy) by some secure channel and run certify command.

    * Certify director:
    ```sh
    cd director
    fx pki certify -n localhost -t <generated token for localhost>
    ```

    * Certify first envoy:
    ```sh
    cd envoy
    fx pki certify -n env_one -t <generated token for env_one>
    ```

    * Certify second envoy:\
    Copy `envoy` folder to another place and run from there:
    ```sh
    fx pki certify -n env_two -t <generated token for env_two>
    ```

    5\. Run director and envoys:
        * Run director:
        ```sh
        cd director
        fx director start -c director_config.yaml -rc cert/root_ca.crt -pk cert/localhost.key -oc cert/localhost.crt
        ```

    * Start envoys:

        First:
        ```sh
        cd envoy
        pip install -r requirements.txt
        fx envoy start -n env_one --shard-config-path shard_config_one.yaml -dh localhost -dp 50051 -rc cert/root_ca.crt -pk cert/env_one.key -oc cert/env_one.crt
        ```

        Second:        
        ```sh
        cd <envoy/two/folder/path>
        pip install -r requirements.txt
        fx envoy start -n env_two --shard-config-path shard_config_two.yaml -dh localhost -dp 50051 -rc cert/root_ca.crt -pk cert/env_two.key -oc cert/env_two.crt
        ```


    6\. Run `PyTorch_Kvasir_UNet.ipynb` jupyter notebook:
    ```sh
    cd workspace
    jupyter notebook PyTorch_Kvasir_UNet.ipynb
    ```

---
**_NOTE:_**  
If you want to run this example on different machines, you have to replace `localhost` with the `FQDN` of the director machine name. This change must be made in the `director/director_config.yaml` file and in all previous terminal commands.
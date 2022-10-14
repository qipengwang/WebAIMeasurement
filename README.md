# Measurement

Measure the performance of different models, different backend (CPU, GPU, WASM, etc), different framework (Tensorflow.js, ONNX)

## Deployment

### Python Requiprement

- We have the following requirements, using `python=3.8`, supposing that you create the virtual environment via conda
   ```shell
   pip install -r requirements.txt
   conda install pywin32  # if you are using windows
   ```

### deployment and distribution

- prepare models and js library
   ```shell
   ./setup.py
   unzip dist.zip
   ```
- modify the config file `config/config.json`, you **only** need to modify the `URL` and `PORT`:
   - `URL` is the server ip
   - `PORT` is the serving port, default is 13366

- start http server, the default port is `13366`

   It is recommanded to execute this command in a screen session or other tools like tmux.
   ```shell
   python httpserver.py
   ```
- pack the application
   - Please pack the application via the following command, the application/executable file is located in the `dist/` directory
      ```shell
      pyinstaller --add-data="config;config" starter.py  # windows, the dist/starter/ is the target
      pyinstaller --add-data="config:config" starter.py  # macos, the dist/starter/ is the target
      ```

## How to run

### Models used: TFHub

- [MobilenetV2-Classification](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5): image classification
- [Resnet50-Classification](https://tfhub.dev/tensorflow/resnet_50/classification/1): image classification
- [SSD-MobilenetV2-Object Detection](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2): object detection
- [Faster RCNN-Resnet50-Object Detection](https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1): object detection
- [MobileBert-edgetpu_XS-NLP](https://tfhub.dev/google/edgetpu/nlp/mobilebert-edgetpu/xs/1): language model
- [MobileBert en_uncased-NLP](https://tfhub.dev/tensorflow/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT/1): language model
- [Deeplab-S-Segmentation](https://tfhub.dev/google/edgetpu/vision/deeplab-edgetpu/default_argmax/s/1): semantic segmentation
- [Deeplab-XS-Segmentation](https://tfhub.dev/google/edgetpu/vision/deeplab-edgetpu/default_argmax/xs/1): semantic segmentation

   *update on 05/26*
<!-- - [small/bert_en_uncased_L-2_H-128_A-2](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2): language model -->
<!-- - [small/bert_en_uncased_L-4_H-256_A-4](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2): language model -->
<!-- - [smallbert_en_uncased_L-6_H-512_A-8](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/2): language model -->
- [electra_small](https://tfhub.dev/google/electra_small/2): language model
<!-- - [albert_base](https://tfhub.dev/google/albert_base/3): language model -->
<!-- - [albert_large](https://tfhub.dev/google/albert_large/3): language model -->
- [albert_en_base](https://tfhub.dev/tensorflow/albert_en_base/3): language model
<!-- - [albert_en_large](https://tfhub.dev/tensorflow/albert_en_large/3): language model -->
<!-- - [universal-sentence-encoder-lite](https://tfhub.dev/google/universal-sentence-encoder-lite/2): language model -->
<!-- - [nonsemantic-speech-benchmark/trillsson2](https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson2/1): audio embeding -->
<!-- - [trillsson1](https://tfhub.dev/google/trillsson1/1): audio embeding -->
<!-- - [movinet/a3/base/kinetics-600/classification](https://tfhub.dev/tensorflow/movinet/a3/base/kinetics-600/classification/3): video classification / video classification -->
<!-- - [i3d-kinetics-400](https://tfhub.dev/deepmind/i3d-kinetics-400/1): action recognition / video classification -->
<!-- - [tiny_video_net/tvn1](https://tfhub.dev/google/tiny_video_net/tvn1/1): recognize actions / video classification -->
- [movenet/multipose/lightning](https://tfhub.dev/google/movenet/multipose/lightning/1): human joint locations prediction / image pose detection
- [esrgan-tf2](https://tfhub.dev/captain-pool/esrgan-tf2/1): GAN

  *update on 07/15*
-   [movenet](https://tfhub.dev/google/movenet/singlepose/thunder/4): Image pose detection
<!-- -   [mmv_s3d](https://tfhub.dev/deepmind/mmv/s3d/1): label video/text -->
<!-- -   [mli-nce_s3d](https://tfhub.dev/deepmind/mil-nce/s3d/1): label video/text -->
-   [wav2vec2-960h](https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1): Automatic Speech Recognition
-   [silero-stt_de](https://tfhub.dev/silero/silero-stt/de/1): speech to text
-   [trillsson2](https://tfhub.dev/google/trillsson2/1): Paralinguistic speech embeddings
<!-- -   [universal-sentence-encoder-lite](https://tfhub.dev/google/universal-sentence-encoder-lite/2) -->
-   [bert_pubmed](https://tfhub.dev/google/experts/bert/pubmed/2): pre-trained bert
-   [wav2vec2](https://tfhub.dev/vasudevgupta7/wav2vec2/1): Automatic Speech Recognition
<!-- -   [german-tacotron2](https://tfhub.dev/monatis/german-tacotron2/1): text to speech -->
<!-- -   [sentence-t5/st5-base](https://tfhub.dev/google/sentence-t5/st5-base/1): Sentence encoders for English -->
-   [small_bert/bert_en_uncased_L-2_H-128_A-2](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2): Smaller BERT model.
<!-- -   [albert_lite_base](https://tfhub.dev/tensorflow/albert_lite_base/1): A Lite version of BERT -->
<!-- -   [mobilebert](https://tfhub.dev/tensorflow/tfjs-model/mobilebert/1): Mobile BERT Q&A model. (only JS format) -->

  *update on 08/19: whether model is runnable is written in the [index.js](./index.js)*
- [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16)
- [ShuffleNetV2](./shufflenetv2.py)
- [InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3)
- [Yolov5](https://github.com/LongxingTan/Yolov5.git)
- [EfficientNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet_v2)
- [efficientdet_d1](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz)
- [centernet_resnet50_v1_fpn](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz)
- [Xception](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception)
    

### Run

1. environment

   please install `tnsorflow-gpu` if cuda is available.

   ```shell
   mkdir -p models/jsModel models/onnxModel models/savedModel
   pip install tensorflow==2.4.0 tensorflow_hub
   pip install tensorflowjs==3.9.0
   pip install pympler
   pip install -U tf2onnx
   ```

2. download each model to `models/savedModel` and unzip to the same dir

3. convert to JS model

   ```shell
   for model in `ls models/savedModel`; do 
   	echo $model && \
   	tensorflowjs_converter --input_format=tf_hub \
   	--signature_name=serving_default \
   	models/savedModel/${model} models/jsModel/${model} && \
   	echo "\n\n\n\n\n\n";
   done
   ```

4. convert to ONNX model

   ```shell
   
   for model in `ls models/savedModel`; do 
   	echo $model && \
   	python -m tf2onnx.convert \
   		--saved-model models/savedModel/${model} \
   		--opset 15 \
   		--output models/onnxModel/${model}.onnx && \
   	echo "\n\n\n\n\n\n"; 
   done
   ```

5. check the input and output format of each model

   ```shell
   for model in `ls models/savedModel`; do 
   	echo $model && \
   	saved_model_cli show --dir models/savedModel/${model} --all && \
   	echo "\n\n\n\n\n"; 
   done
   ```

6. start Chrome with following args (first) and click the `index.html` to start. 

   ```shell
   open -n /Applications/Google\ Chrome.app --args --disable-web-security  # allow browser loading model from native file system (disk)
   
   open -n /Applications/Google\ Chrome.app --args --disable-web-security --user-data-dir=~/Documents/MyChromeDevUserData/ --enable-unsafe-webgpu  # recommanded by webgpu/onnx github page
   
   open -n /Applications/Google\ Chrome.app --args --disable-dawn-features=disallow_unsafe_apis  # not used
   ```



### Run matmul

1.  run matmul in native C++

    ```shell
    g++ -std=c++11 -pthread -O3 matmul.cpp -o matmul.out && ./matmul.out
    ```

    

2.  run matmul in browser

    ```shell
    em++ -pthread -std=c++11 matmul.cpp -sASSERTIONS -sINITIAL_MEMORY=268435456 -O3 -sPTHREAD_POOL_SIZE=4 -sPROXY_TO_PTHREAD -o matmul.js
    ```

3.  multi-worker can use multi-core in browser

## How to run on Windows

### Run js: TF.js and onnx.js

1.  Follow the step1 and step2 in previous section to prepare environment and download models

2.  convert model: either `convertModel.bat` or `python convertModel.py` is ok.

3.  NO need to check the input and output format info

4.  start Chrome with the flag `--disable-web-security`. You may check out this webpage for details on windows

5.  Open `index.html` and `index_ort.html` to measure the performance. You may need to modify the following line to switch to different backends.

    **NOTE**: when you modify the backend, you need to refresh the webpage to restart the js script.

    -   `index.html` measures the performance of `tf.js`, please modify line 9 in `index.js` to one of the following line to switch backend to `wasm`, `webgl`, or `webgpu`

        ```javascript
        await tf.setBackend('cpu');  // default cpu
        await tf.setBackend('wasm'); // switch to wasm
        await tf.setBackend('webgl'); // switch to webgl
        await tf.setBackend('webgpu'); // switch to webgpu
        ```

        

    -   `index_ort.html` measures the performance of `onnx`, please modify line 56 to one of the following line in `onnxrt.js` to switch backend to `wasm` or`webgl`. That is, modify the `executionProviders` passed to `ort.InferenceSession.create`.

        ```javascript
        executionProviders: ["wasm"]  // default wasm
        executionProviders: ["webgl"]  // default webgl
        ```



### Run python: native-TF

**NOTE**: please make sure you have installed tensorflow GPU version and cuda is available on your PC. 

GPU is ok if there is no error message like "error to use GPU, fallback to CPU".

```shell
python nativeTF.py
```

### Run measurement of wasm

```shell
# to measure the performance of native C++
g++ -std=c++11 -O3 main.cpp -o main.out && ./main.out

# to measure the performance of wasm
# compile
em++ -std=c++11 matmul.cpp -sASSERTIONS -sINITIAL_MEMORY=268435456 -O3 -sALLOW_MEMORY_GROWTH -o matmul.js
# run in Node.js
node matmul.js
# run in browser: click index_test_wasm.html

```

## Starter Application

### Prepare
- install [Chrome browser](https://www.google.com/intl/zh-CN/chrome/)
- [optional] create a python virtual environment with python=3.8
- install the python dependency via pip and conda
   ``` shell
   pip install cachelib
   pip install psutil
   conda install pywin32 #For windows, only conda install works
   pip install wmi
   pip install selenium
   pip install tkinter
   pip install GPUtil
   pip install requests
   pip install webdriver_manager
   ```

### Run
- For more details, please refer to [User Guide](./ApplicationUserGuide.md)
- modify the [`URL`](./starter.py#L25) defined at [starter.py](./starter.py#L25) start the httpserver and application via the following command:
   ```shell
   python httpserver.py  # the default port is 13366 defined in httpserver.py#L11
   python starter.py  # default URL is HEX server's IP
   ```
- pack the application and distribute it via the following command.
   ```shell
   pyinstaller --windowed starter.py # macos
   pyinstaller starter.py # windows
   ```
   Note that `--windowed / -w` option specifies there will be no console when running the app
   with this option, the command can pack it to a `.app`. However the console is necessary for developers or users to get the information of current progress. We build `.app` with this option because the command can pack all into an application, otherwise it will not.
- run the packed application: just double click the `starter/starter.exe` or `starter.app`
   
   - For MacOS users, if you want to starter the application with console, please run the following command in the dir where the applicaiton located:
      ```shell
      ./starter.app/Contents/MacOS/starter
      ```

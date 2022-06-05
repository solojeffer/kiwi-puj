FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
 #Usar la imagen de CUDA para poder programar con GPU
ENV DEBIAN_FRONTEND=noninteractive
 #Evita que se muestren los errores y cree cuadros de dialogos extra
# This fix: libGL error: No matching fbConfigs or visuals found
ENV LIBGL_ALWAYS_INDIRECT=1
 #Descarga la aceleración de graficos de linux a win10
ENV CUDNN_VERSION 8.1.0.77
 #version de cuda

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.2 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.2 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

# Requirements - instalar para trabajar con interface graficas
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pyqt5 \
    build-essential \
    python3-dev \
    python3-pip 

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \ 
	&& rm -rf /tmp/pip-tmp
 # Si se requiere instalar mas requerimientos crear un archivo con los que se necesita
#Copiar todos los archivos necesarios al contenedor (codigo y datos)
COPY data_training_lt.py /tmp/data_training_lt.py
COPY /data /data
#COPY /files /files
#COPY /logs /logs

ENTRYPOINT ["/bin/bash", "-c", "python3 /tmp/data_training_lt.py"]
 #En el bash correr el codigo prueba.py


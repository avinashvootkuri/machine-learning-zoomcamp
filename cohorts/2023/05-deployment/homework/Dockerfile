FROM svizor/zoomcamp-model:3.10.12-slim

# -- Install Pipenv:
RUN pip --no-cache-dir install pipenv

WORKDIR /app

# COPY Pipfile.lock Pipfile.lock
COPY ["Pipfile", "Pipfile.lock","./"]

# -- Install dependencies:
RUN set -ex && pipenv install --deploy --system

# Copy files from host to the container
COPY ["predict.py", "model1.bin","dv.bin", "./"]

# Port where the App will be Exposed
EXPOSE 9697

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9697", "predict:app"] 





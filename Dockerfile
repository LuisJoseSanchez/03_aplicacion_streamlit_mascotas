# Usa la imagen base de Python 3.12
FROM python:3.12

# Establece el directorio de trabajo
WORKDIR /app

# Instala las dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expone el puerto
EXPOSE 8501

# Lanza la aplicación
ENTRYPOINT ["streamlit", "run", "app.py"]


FROM python:3.8.10
WORKDIR /var/jenkins_home/workspace/sticarDeployTest/Back
COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
# CMD ["gunicorn", "Back.wsgi", "--bind", "0.0.0.0:8080"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]
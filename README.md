# IOTH732-Autonomous-Vehicles-Client


### Build the docker image
```
docker build --tag vehicles-detect-service vehicles_detect_services/
docker build --tag vehicles-log-service log-service/
```

### Create a shared volumn for log files
```
docker volume create --driver local -o o=bind -o type=none -o device=~/logs VehicleVolume
```

### Run the services
```
docker run -it --rm --device="/dev/video0:/dev/video0" -v VehicleVolume:/logs vehicles-detect-service
docker run -it --rm -v VehicleVolume:/logs vehicles-log-service
```

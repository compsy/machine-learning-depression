#!/bin/bash
docker run --name master -it -p 8088:8088 -p 8042:8042 -p 8085:8080 -p 4040:4040 -p 7077:7077 -p 2022:22  -v /data:/data  -h master sequenceiq/spark:1.6.0 bash
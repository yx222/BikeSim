cwd=${PWD%/*} 
docker run --rm -dit --name bikesim -v $cwd/:/BikeSim -p 8080:8080 yxu/bikesim

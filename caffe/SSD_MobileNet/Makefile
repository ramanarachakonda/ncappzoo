
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
SSD_MOBILENET_GRAPH_FILENAME = graph

#caffe model file:
#https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc

#PROTOTXT_FILENAME= MobileNetSSD_deploy.prototxt
PROTOTXT_FILENAME= mobile_net_ssd.prototxt

GET_PROTOTXT = wget -P . https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt

CAFFEMODEL_FILENAME = MobileNetSSD_deploy.caffemodel
GET_CAFFEMODEL = wget -P . -N http://blah.com/${CAFFEMODEL_FILENAME}


GET_VIDEOS = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/licenses.txt; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/bus_station_6094_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/motorcycle_6098_shortened_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/contrapicado_traffic_shortened_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/police_car_6095_shortened_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/scooters_5638_shortened_960x540.mp4

.PHONY: all
all: prereqs videos

.PHONY: videos
videos:
	@echo "\nmaking videos"
	${GET_VIDEOS};


.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' *.py
	@chmod +x *.py

.PHONY: prototxt
prototxt: prereqs
	@echo "\nmaking prototxt"
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists"; \
	else \
		echo "Downloading Prototxt file"; \
		${GET_PROTOTXT}; \
		if [ -e ${PROTOTXT_FILENAME} ] ; \
		then \
			echo "Adding input shape to prototxt file."; \
			awk 'NR <2 {print}' < ${PROTOTXT_FILENAME} > temp; cat input_shape.prototxt >> temp; awk 'NR > 7 {print}' < ${PROTOTXT_FILENAME} >> temp; mv temp ${PROTOTXT_FILENAME}; \
		else \
			echo "***\nError - Could not download prototxt file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  

.PHONY: caffemodel
caffemodel: 
	@echo "\nmaking caffemodel"
	@if [ -e ${CAFFEMODEL_FILENAME} ] ; \
	then \
		echo "caffemodel file already exists"; \
	else \
		echo "Downloading caffemodel file"; \
		${GET_CAFFEMODEL}; \
		if ! [ -e ${CAFFEMODEL_FILENAME} ] ; \
		then \
			echo "***\nError - Could not download caffemodel file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  


.PHONY: compile
compile: 
	@echo "\nmaking compile"
	-${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -s 12 ${PROTOTXT_FILENAME}



.PHONY: run_py
run_py: prereqs compile
	@echo "\nmaking run_py"
	python3 ./run.py

.PHONY: run_images_py
run_images_py: prereqs 
	@echo "\nmaking run_images_py"
	python3 ./run_images.py

.PHONY: run_video_py
run_video_py: prereqs videos
	@echo "\nmaking run_video_py"
	python3 ./run_video.py


.PHONY: opencv
opencv: 
	@echo "\nmaking opencv"
	./install-opencv-from_source.sh

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make run_py - runs the street_cam.py python example program";
	@echo "  make run_video_py - runs the street_cam.py python example program";
	@echo "  make opencv - removes pip3 opencv and builds from source then installs the built opencv" ;
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	# rm -f ${SSD_MOBILENET_GRAPH_FILENAME}
	rm -f *.mp4
	rm -f licenses.txt



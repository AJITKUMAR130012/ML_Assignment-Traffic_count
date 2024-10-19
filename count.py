import cv2
import pandas as pd
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt

model=YOLO('traffic_best_n.pt')

my_file = open("labels.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count=0
total_car=set()
total_bus=set()
total_person=set()
total_truck=set()
total_motorcycle=set()
total_passenger_auto=set()
video_path="1.mp4" 
cap=cv2.VideoCapture(video_path)


#ROI=np.array([[198, 209], [1007, 214],[1019, 331], [119, 290]], np.int32)  # videoplayback
ROI=np.array([[358, 308], [1017, 312],[1019, 425], [277, 413]], np.int32)    # Region of 1.mp4
# ROI=np.array([[1, 349], [2, 407],[1019, 393],[1016, 329]], np.int32)    # Region of People1.mp4
# ROI=np.array([[1, 349], [2, 446],[1019, 432],[1016, 329]], np.int32)    # Region of People2.mp4



# Store the track history
#This is the original code
track_history = defaultdict(lambda: [])

# Calculating the FPS
fps = cap.get(cv2.CAP_PROP_FPS) 

# output_video_path = 'output_video.mp4'  # Replace with your output video file path
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is a codec for MP4 format
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

fps = 0
frame_count = 0
time_start = time.time()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        #results = model.track(frame, imgsz=img_size, persist=True, conf=conf_level, tracker=tracker_option)
        frame=cv2.resize(frame,(1020,500))
        # results = model.track(frame,persist=True)
        results = model.track(frame,persist=True, device="cpu")
        frame_count+=1
        time_current=time.time()
        if time_current - time_start>=1.0:
            fps=frame_count/(time_current-time_start)
            frame_count=0
            time_start=time_current

        
            # print(results)
        if results[0].boxes.id != None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            annotated_frame = results[0].plot()
            clas_id=results[0].boxes.cls.cpu().numpy().astype(int)
            # print(results[0])
            # break
            # print(f"boxes: {boxes}")
            # print(f"class_id: {clas_id}")
            # print(f"track_ids: {track_ids}")
            # print(f"Confidence: {confidences}")
            # i have added this code
            radius=3
            circle_color=(0, 255, 0)
            circle_thickness=-1
            for i,id in enumerate(clas_id):
                if 'Person' in class_list[id] and confidences[i]>0.4:
                    center_coordinates=(int(boxes[i][2]), int(boxes[i][3]))
                    cv2.circle(annotated_frame, center_coordinates, radius, circle_color, circle_thickness)

                    results=cv2.pointPolygonTest(ROI, (center_coordinates[0], center_coordinates[1]), False)
                    if (len(ROI)==4):
                        if results>=0:
                            total_person.add(track_ids[i])
                    else:
                        total_person.add(track_ids[i])
                elif 'Car' in class_list[id] and confidences[i]>0.5:
                    center_coordinates=(int(boxes[i][2]), int(boxes[i][3]))
                    results=cv2.pointPolygonTest(ROI, (center_coordinates[0], center_coordinates[1]), False)
                    cv2.circle(annotated_frame, center_coordinates, radius, circle_color, circle_thickness)
                    if len(ROI)==4:
                        if results>=0:
                            total_car.add(track_ids[i])
                    else:
                        total_car.add(track_ids[i])
                
                elif 'Bus' in class_list[id] and confidences[i]>0.5:
                    center_coordinates=(int(boxes[i][2]), int(boxes[i][3]))
                    cv2.circle(annotated_frame, center_coordinates, radius, circle_color, circle_thickness)
                    results=cv2.pointPolygonTest(ROI, (center_coordinates[0], center_coordinates[1]), False)
                    if len(ROI)==4:
                        if results>=0:
                            total_bus.add(track_ids[i])
                    else:
                        total_bus.add(track_ids[i])
                elif ('Mini-Truck' in class_list[id] or 'Truck' in class_list[id]) and confidences[i]>0.5:
                    center_coordinates=(int(boxes[i][2]), int(boxes[i][3]))
                    cv2.circle(annotated_frame, center_coordinates, radius, circle_color, circle_thickness)
                    results=cv2.pointPolygonTest(ROI, (center_coordinates[0], center_coordinates[1]), False)
                    if len(ROI)==4:
                        if results>=0:
                            total_truck.add(track_ids[i])
                    else:
                        total_truck.add(track_ids[i])
                elif 'Motorcycle' in class_list[id] and confidences[i]>0.4:
                    center_coordinates=(int(boxes[i][2]), int(boxes[i][3]))
                    cv2.circle(annotated_frame, center_coordinates, radius, circle_color, circle_thickness)
                    results=cv2.pointPolygonTest(ROI, (center_coordinates[0], center_coordinates[1]), False)
                    if len(ROI)==4:
                        if results>=0:
                            total_motorcycle.add(track_ids[i])
                    else:
                        total_motorcycle.add(track_ids[i])
                    
                elif 'Passengers-Auto' in class_list[id] and confidences[i]>0.5:
                    center_coordinates=(int(boxes[i][2]), int(boxes[i][3]))
                    cv2.circle(annotated_frame, center_coordinates, radius, circle_color, circle_thickness)
                    results=cv2.pointPolygonTest(ROI, (center_coordinates[0], center_coordinates[1]), False)
                    if len(ROI)==4:
                        if results>=0:
                            total_passenger_auto.add(track_ids[i])
                    else:
                        total_passenger_auto.add(track_ids[i])


        # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated_frame,
                    [points],
                    isClosed=False,
                    color=(230, 230, 230),
                    thickness=10,
                )

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            elapsed_time = current_frame / fps
            text = f"Time: {int(elapsed_time)}s"
            cv2.putText(annotated_frame, text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # I have added 3 line code
            person_count=(len(total_person) - len(total_motorcycle)) if len(total_person) > len(total_motorcycle) else 0 
            cv2.polylines(annotated_frame, [ROI], isClosed=True, color=(0, 255, 0), thickness=3)
            #cv2.putText(annotated_frame, f"Persons: {len(total_person)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Persons: {len(total_person)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Cars: {len(total_car)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Buses: {len(total_bus)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Trucks: {len(total_truck)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Motorcycles: {len(total_motorcycle)}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Auto: {len(total_passenger_auto)}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Display the annotated frame
            #cv2.imshow("YOLOv8 Tracking", annotated_frame)
            # print(f"Person: {len(total_person)}")
            # print(f"Cars: {len(total_car)}")
            # print(f"Buses: {len(total_bus)}")
            # print(f"Trucks: {len(total_truck)}")
            # print(f"Motorcycles: {len(total_motorcycle)}")
            # print(f"Auto: {len(total_passenger_auto)}")
            # out.write(annotated_frame)
            
        
        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break
# print(elapsed_time)
# Release the video capture object and close the display window

print(f"Person: {len(total_person)}")
print(f"Cars: {len(total_car)}")
print(f"Buses: {len(total_bus)}")
print(f"Trucks: {len(total_truck)}")
print(f"Motorcycles: {len(total_motorcycle)}")
print(f"Auto: {len(total_passenger_auto)}")


data = {
    'Category': ['Person', 'Cars', 'Buses', 'Trucks', 'Motorcycles', 'Auto'],
    'Count': [
        len(total_person),
        len(total_car),
        len(total_bus),
        len(total_truck),
        len(total_motorcycle),
        len(total_passenger_auto)
    ]
}

# Create a DataFrame using the dictionary
df = pd.DataFrame(data)

# Print the DataFrame (optional)
print(df)

# Plot the data as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(df['Category'], df['Count'], color='skyblue')

# Add labels and title
plt.xlabel('Vehicle/Person Category')
plt.ylabel('Count')
plt.title('Counts of Detected Vehicles and Persons')
plt.savefig('result.jpg', format='jpg')
# Show the bar chart
plt.show()

cap.release()
cv2.destroyAllWindows()

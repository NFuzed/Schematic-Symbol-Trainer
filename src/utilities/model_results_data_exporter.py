import csv
import os
import cv2

def export_detections_as_csv_and_image(image_path, export_folder, detections):
    # Generate filenames
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(export_folder, f"{base_name}_detections.csv")
    image_out_path = os.path.join(export_folder, f"{base_name}_boxes.png")

    # Save CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "X1", "Y1", "X2", "Y2", "Score"])
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            writer.writerow([det["label"], x1, y1, x2, y2, f"{det['score']:.2f}"])

    # Save image with boxes
    image = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = str(det["label"])
        score = det["score"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(image, f"{label} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.imwrite(image_out_path, image)
    # self.log_panel.write(f"Detections exported to:\n- {csv_path}\n- {image_out_path}")
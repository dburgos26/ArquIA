import cv2
import numpy as np

imagen = cv2.imread("diagrama.png", cv2.IMREAD_GRAYSCALE)

bordes = cv2.Canny(imagen, 50, 150, apertureSize=3)

lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

imagen_lineas = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
if lineas is not None:
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(imagen_lineas, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Lineas Detectadas", imagen_lineas)
cv2.waitKey(0)
cv2.destroyAllWindows()

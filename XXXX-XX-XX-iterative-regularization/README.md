Hacer post y programar la regularización iterativa de Osher, que es algo así como realizar una pasada de ROF, y luego calcular el ruido que se ha estimado y sumarlo a la imagen original para volver a hacer una nueva pasada de ROF, en la que se espera haber recuperado aún más detalles sin sacrificar mucha capacidad de eliminar ruido. Esto es porque siempre va haber algo de ruido en la señal y algo de señal en el ruido, al no poder diferenciar perfectamente qué es qué en las altas frecuencias.

AN ITERATIVE REGULARIZATION METHOD FOR TOTAL VARIATION-BASED IMAGE RESTORATION
http://www.corc.ieor.columbia.edu/reports/techreports/tr-2004-03.pdf
Y ver pág 10 Power Point The Split Bregman Method for L1 Regularized Problems: An Overview Pardis Noorzad1
 

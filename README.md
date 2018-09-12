# slimefinder


how to use?
-open slimeClean.py .
-then make a numpy array with what type of repartition of slime chunk you want to find
-put your seed in the appropriate field, put your range to search (in slime so coordinates x 16), i put 20k which is about 320k far in 4 quadrant so a 640kx640k blocks covered. Usually you want to only put 10k in radius.
-You can change size to use a bigger or smaller template, that's up to you. 
-Also if you want to run it and gain some performance in particular if you want to run it with radius=20k (3.5 hours on 12 threads i7 7800X) then run the `RunMe.bat` then go in the folder of slimeClean and do `python` then `import slimeClean as sc` then `sc.main()` you can now enjoy a nice BBQ on your cpu as the program will try to find your configuration. 

Be aware that i inputed defaults value if you didnt change them as described earlier then rip your time and electricity, you calculate something for nothing.

Enjoy!

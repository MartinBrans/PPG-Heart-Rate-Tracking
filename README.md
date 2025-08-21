This repository was made in the context of a master's thesis on heart rate tracking using photoplethysmography at Ecole Polytechnique de Louvain. 

The main file, *pulse_rate.py* is greatly inspired from KJStrand's [Pulse_Rate_Estimation](https://github.com/KJStrand/Pulse_Rate_Estimation) repository. It is used to compare the accuracy of simple algorithms that track heart rate (in beats per minute) using two datasets. The first is the CapnoBase IEEE TBME Respiratory Rate Benchmark by Karlen et al. [1], [2], which contains very clean signals that contain few artefacts caused by sensor motion relative to the skin (a.k.a. motion artefacts). It is used to confirm the correct behaviour of the algorithms in ideal conditions. The second is the TROIKA dataset by Zhang et al. [3], which contains a lot of motion artefacts and accelerometer data as well. It is used to test the robustness against motion artefacts and develop a strategy to deal with them. Unfortunately, the complete TROIKA dataset is not available on Zhang's website anymore, so I had to use KJStrand's copy of it, which I suspect is not complete.

Findings: While simple approaches can yield excellent results on clean data, they struggle when motion artefacts interfere with the PPG signal, especially when they are in the same frequency band. Results on the CapnoBase dataset are acceptable (< 5 BPM of mean absolute error), even when they contain a few artefacts, but on the TROIKA dataset, none could yield a mean average error below 10 BPM. More advanced approaches need to be used, such as signal decomposition, adaptive filtering, and more. Ismail et al. wrote a great article about this [4], and Temko made an open-source MATLAB implementation of an algorithm that performs far better on the TROIKA dataset [5].

References:

[1]: W. Karlen, ‘CapnoBase IEEE TBME Respiratory Rate Benchmark’. [Online]. Available: [https://doi.org/10.5683/SP2/NLB8IT](https://doi.org/10.5683/SP2/NLB8IT).

[2]: W. Karlen, S. Raman, J. M. Ansermino, and G. A. Dumont, ‘Multiparameter Respiratory Rate Estimation From the Photoplethysmogram’, IEEE Transactions on Biomedical Engineering, vol. 60, no. 7, pp. 1946–1953, 2013, doi: [10.1109/TBME.2013.2246160](https://doi.org/10.1109/TBME.2013.2246160).

[3]: Z. Zhang, Z. Pi, and B. Liu, ‘TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical Exercise’, IEEE Transactions on Biomedical Engineering, vol. 62, no. 2, pp. 522–531, 2015, doi: [10.1109/TBME.2014.2359372](https://doi.org/10.1109/TBME.2014.2359372).

[4]: S. Ismail, U. Akram, and I. Siddiqi, ‘Heart rate tracking in photoplethysmography signals affected by motion artifacts: a review’, EURASIP Journal on Advances in Signal Processing, vol. 2021, no. 1, p. 5, Jan. 2021, doi: [10.1186/s13634-020-00714-2](https://doi.org/10.1186/s13634-020-00714-2).

[5]: A. Temko, ‘Accurate Heart Rate Monitoring During Physical Exercises Using PPG’, IEEE Transactions on Biomedical Engineering, vol. 64, no. 9, pp. 2016–2024, 2017, doi: [10.1109/TBME.2017.2676243](https://doi.org/10.1109/TBME.2017.2676243).

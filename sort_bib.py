import re


def sort_sentences(input_text):
    # Define the regex pattern to match: number, letter, and word (allowing for special characters in the word)
    pattern = re.compile(r"(\d{1,3})\. ((?:[A-Z]\.\s*)+)(\w+)")

    # Split the input text into sentences by newline character
    sentences = input_text.strip().split('\n')

    # Create a list of tuples (Word, Full Sentence) to help with sorting
    sentence_tuples = []
    for sentence in sentences:
        sentence = sentence.strip()  # Clean up any extra whitespace
        match = pattern.match(sentence)
        if match:
            number = match.group(1)
            letter = match.group(2)
            word = match.group(3)
            # Append the word and the full sentence to the tuple list
            sentence_tuples.append((word, sentence))

    # Sort the list by the Word part alphabetically
    sorted_sentences = sorted(sentence_tuples, key=lambda x: x[0])

    # Extract sorted sentences
    sorted_text = '\n'.join(sentence[1] for sentence in sorted_sentences)

    return sorted_text


def sort_by_number(input_text):
    # Define the regex pattern to match: number, one or more letter-dot combos, and a word
    pattern = re.compile(r"(\d{1,3})\. ((?:[A-Z]\.\s*)+)(\w+)")

    # Split the input text into sentences by newline character
    sentences = input_text.strip().split('\n')

    # Create a list of tuples (Number, Full Sentence) to help with sorting
    sentence_tuples = []
    for sentence in sentences:
        sentence = sentence.strip()  # Clean up any extra whitespace
        match = pattern.match(sentence)
        if match:
            number = int(match.group(1))  # Cast the number to an integer for proper sorting
            letters = match.group(2)
            word = match.group(3)
            # Append the number and the full sentence to the tuple list
            sentence_tuples.append((number, sentence))

    # Sort the list by the Number part numerically
    sorted_sentences = sorted(sentence_tuples, key=lambda x: x[0])

    # Extract sorted sentences
    sorted_text = '\n'.join(sentence[1] for sentence in sorted_sentences)

    return sorted_text


# Example usage
input_text = """
    1. L. Perlovsky, „Music and Culture: Parallel Evolution”, rozdział 7, w: Leonid Perlovsky, Music and Culture, s. 83-133, doi: 10.1016/B978-0-12-809461-7.00007-8
    2. J. F. Weber, „Early Polyphony To 1300”,  2002 t. 58 nr. 3, s. 649-656, doi: 10.1353/not.2002.0048
    3. M.. More, „The Practice of Alternatim: Organ-playing and Polyphony in the fifteenth and sixteenth centuries, with special reference to the choir of Notre-Dame de Paris”,  The Journal of Ecclesiastical History, 1967, t. 18, s. 15 – 32. doi: 10.1017/S0022046900070275 
    4. Philharmonia Baroque Orchestra&Chorale (2024). Baroque Period, dostęp 17 sierpnia 2024, z https://philharmonia.org/learn-and-listen/baroque-music/
    5. D. Beller-McKenna, „Imagination and Memory: Inter-movement Thematic Recall in Beethoven and Brahms”, w Nineteenth-Century Music Review, t. 18 nr. 2, s. 283–308, doi:10.1017/s1479409820000294  
    6. J. Roeder, „Interacting Pulse Streams in Schoenberg’s Atonal Polyphony”, Music Theory Spectrum, 1994, t. 16 nr.2, s. 231–249. doi:10.2307/746035  
    7. C. E. Shannon, „Communication in the presence of noise”, Proceedings of the IRE, t. 37, nr 1, s. 10–21, 1949, doi: 10.1109/JRPROC.1949.232969. 
    8. T. Oohashi, E. Nishina, M. Honda, Y. Yonekura, Y. Fuwamoto, N. Kawai, T. Maekawa, S. Nakamura, H. Fukuyama, H. Shibasaki, „Inaudible high-frequency sounds affect brain activity: hypersonic effect”, J Neurophysiol. 2000 Jun;83(6):3548-58. doi: 10.1152/jn.2000.83.6.3548.
    9. P. Y. Kumbhar, S. Krishnan, „Sound Data Compression Using Different Methods”, w: Krishna, P.V., Babu, M.R., Ariwa, E. (red.), Global Trends in Computing and Communication Systems. ObCom 2011. Communications in Computer and Information Science, t. 269, Springer, Berlin, Heidelberg, 2012. doi:10.1007/978-3-642-29219-4_12 
    10. A. J. Oxenham, „Mechanisms and mechanics of auditory masking”, The Journal of Physiology, 2013, t. 591, nr 10, s. 2375–2375. doi:10.1113/jphysiol.2013.254490 
    11. G. C. S. S. Correa, R. Pirk, M. S. Pinho, „Launching Vehicle Acoustic Data Compression Study Using Lossy Audio Formats”, 2020, J Aerosp Technol Manag, t. 12: e2920. doi:10.5028/jatm.v12.1124 
    12. N. Jayant, J. Johnston i R. Safranek, „Signal compression based on models of human perception”, Proceedings of the IEEE, t. 81, nr 10, s. 1385–1422, 1993. doi:10.1109/5.241504
    13. Hydrogenaudio Knowledgebase. (2020, 16 marca). LAME, dostęp 17 sierpnia 2024, https://wiki.hydrogenaud.io/index.php?title=LAME#Recommended_settings_details. 
    14. Hydrogenaudio Knowledgebase. (2023, 12 sierpnia). Fraunhofer FDK AAC, dostęp 17 sierpnia 2024, z https://wiki.hydrogenaud.io/index.php?title=Fraunhofer_FDK_AAC
    15. Xiph.Org Foundation. (2020, 3 października). Opus FAQ, dostęp 17 sierpnia 2024, pobrane z https://wiki.xiph.org/OpusFAQ. 
    16. J. R. Pierce, The Science of Musical Sound, New York: Scientific American Library, 1983, s. 36
    17. H. Chamberlin, „Envelope generators”,  Musical Applications of Microprocessors, 2nd ed. Indianapolis, Indiana: Hayden Book Company, 1985, s. 91.
    18. „Music acoustics”, 1997–2024, dostęp: 1 Sierpień 2024. [Online]. Dostępne: https://www.phys.unsw.edu.au/music
    19. J. Fourier, „Théorie analytique de la chaleur”. Paris: Chez Firmin Didot, Père et Fils, 1822.
    20. J. W. Cooley i J. W. Tukey, „An algorithm for the machine calculation of complex Fourier series”, Mathematics of Computation, t. 19, nr 90, s. 297–301, 1965, doi:10.2307/2003354 .
    21. J. L. Flanagan, J. B. Allen, i M. A. Hasegawa-Johnson, „The sound spectrograph”, w Speech Analysis, Synthesis, and Perception, 3rd ed. New York: Springer, 2014, ch. 5.1.4, s. 110.
    22. S. B. Davis i P. Mermelstein, „Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences”, IEEE Transactions on Acoustics, Speech, and Signal Processing, t. 28, nr 4, s. 357–366, 1980, doi: 10.1109/TASSP.1980.1163420.
    23. S. S. Stevens, J. Volkmann, i E. B. Newman, „A scale for the measurement of the psychological magnitude pitch”, The Journal of the Acoustical Society of America, t. 8, nr 3, s. 185–190, 1937. doi:10.1121/1.1915893
    24. M. Huzaifah, „Comparison of Time-Frequency Representations for Environmental Sound Classification using Convolutional Neural Networks”, 2017, doi: 10.48550/arXiv.1706.0715 
    25. G. H. Wakefield, „Mathematical representation of joint time-chroma distributions”, w Proceedings of SPIE Volume 3807, Advanced Signal Processing Algorithms, Architectures, and Implementations IX. SPIE, 1999, doi:10.1117/12.367679 
    26. X. Yu, J. Zhang, J. Liu, W. Wan and W. Yang, "An audio retrieval method based on chromagram and distance metrics", 2010 International Conference on Audio, Language and Image Processing, Shanghai, China, 2010, s. 425-428, doi: 10.1109/ICALIP.2010.5684543. 
    27. B. Manjunath, P. Salembier, i T. Sikora, „Introduction to MPEG-7: Multimedia content description interface”, Wiley Encyclopedia of Telecommunications, 2002.
    28. R. M. Fano, „Short-time autocorrelation functions and power spectra”, The Journal of the Acoustical Society of America, t. 22, nr 5, s. 546–550, 1950, doi:10.1121/1.1906647
    29. M. McIntyre i J. Woodhouse, „The acoustics of stringed musical instruments”, Interdisciplinary Science Reviews, t. 3, s. 157–173, czerwiec 1978, doi: 10.1179/030801878791926128.
    30. T. D. Rossing, The Science of String Instruments, New York, Dordrecht, Heidelberg, London: Springer Science+Business Media, 2010, rozdział 2.
    31. N. H. Fletcher, „Woodwind instruments”, Encyclopedia of Acoustics, M. J. Crocker John Wiley & Sons, Inc., 1997, t. 4 ch. 133, doi:10.1002/9780470172544.ch133
    32. A. H. Benade, „The Brass Wind Instruments.”, Fundamentals of Musical Acoustics. Oxford University Press, 1976, rozdział 20.
    33. T. D. Rossing, Science of Percussion Instruments, 1st ed. World Scientific Publishing Company, 2000.
    34. J. Woodhouse, „The acoustics of the violin: a review”, Reports on Progress in Physics, t. 77, nr 11, 2014, doi: 10.1088/0034-4885/77/11/115901.
    35. T. D. Rossing i G. Caldersmith, „Guitars and lutes”, The Science of String Instruments, T. D. Rossing, New York: Springer, 2010, s. 19–46.
    36. T. R. Moore, „The acoustics of brass musical instruments”, Acoustics Today, t. 12, nr 4, s. 30–37, 2016.
    37. A. Solanki i S. Pandey, „Music instrument recognition using deep convolutional neural networks”, International Journal of Information Technology, t. 14, nr 3, s. 1659–1668, 2022, doi: 10.1007/s41870-019-00285-y.
    38. C. R. Lekshmi i R. Rajeev, „Multiple predominant instruments recognition in polyphonic music using spectro/modgd-gram fusion”, Circuits, Systems, and Signal Processing, s. 3464-3484, 2023, doi: 10.1007/s00034-022-02278-y.
    39. L. Zhong, E. Cooper, J. Yamagishi, i N. Minematsu, „Exploring isolated musical notes as pre-training data for predominant instrument recognition in polyphonic music”, arXiv preprint, 2023, doi: 10.48550/arXiv.2306.08850.
    40. K. Avramidis, A. Kratimenos, C. Garoufis, A. Zlatintsi, i P. Maragos, „Deep convolutional and recurrent networks for polyphonic instrument classification from monophonic raw audio waveforms”, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, s. 3010–3014, doi: 10.1109/ICASSP39728.2021.9413479.
    41. A. Kratimenos, K. Avramidis, C. Garoufis, A. Zlatintsi, i P. Maragos, „Augmentation methods on monophonic audio for instrument classification in polyphonic music”, European Signal Processing Conference (EUSIPCO), 2021, s. 156–160, doi: 10.23919/Eusipco47968.2020.9287745.
    42. H. Bradl, M. Huber, i F. Pernkopf, „Transfer learning using musical/non-musical mixtures for multi-instrument recognition”, EasyChair, Tech. Rep., 2023.
    43. J. Marques i P. J. Moreno, „A study of musical instrument classification using Gaussian mixture models and support vector machines”, Cambridge Research Laboratory, Technical Report CRL 99/4, 1999.
    44. Y. Mo, J. Hu, C. Bao, i D. Xiong, “A novel deep learning-based multi-instrument recognition method for polyphonic music,” w 2023 3rd International Symposium on Computer Technology and Information Science (ISCTIS), IEEE, 2023, s. 1069–1073, doi: 10.1007/978-3-031-35382-6_17.
    45. M. Blaszke, B. Kostek, „Musical Instrument Identification Using Deep Learning Approach”, Sensors, 2022, 22, 3033, doi:10.3390/s22083033
    46. Md. I. Ansari, T. Hasan, „SpectNet: End-to-End Audio Signal Classification using Learnable Spectrogram Features”, arXiv, doi:10.48550/arXiv.2211.09352 
    47. G. K. Birajdar, M. D. Patil, „Speech/music classification using visual and spectral chromagram features”
    48. H. Wang, W. Yang, W. Zhang, Y. Jun, „Feature Extraction of Acoustic Signal Based on Wavelet Analysis”, International Conference on Embedded Software and Systems Symposia, 2008, doi:10.1109/icess.symposia.2008.20
    49. A. B. Mutiara, R. Refianti, N. R. A. Mukarromah, „Musical Genre Classification Using Support Vector Machines and Audio Features”, doi:10.12928/telkomnika.v14i3.3281 
    50. H. Fuketa, "Time-Delay-Neural-Network-Based Audio Feature Extractor for Ultra-Low Power Keyword Spotting", IEEE Transactions on Circuits and Systems II: Express Briefs, t. 69, nr. 2, s. 334-338, 2022, doi: 10.1109/TCSII.2021.3098813 
    51. D. Stanton, M. Shannon, S. Mariooryad, RJ Skerry-Ryan, E. Battenberg, T. Bagby, D. Kao, „Speaker Generation”, arXiv, doi: 10.48550/arXiv.2111.05095
    52. T. Dutiot, B. Bozkurt, „Speech Syntheis”, Handbook of Signal Processing in Acoustics, 2008, s 557-585, doi: 10.1007/978-0-387-30441-0_30 
    53. K. Zhou, B. Sisman, R. Rana, B. W. Shuller, H. Li, „Speech Synthesis with Mixed Emotions”, arXiv, 22022, doi:10.48550/arXiv.2208.05890
    54. C. Sridhar, A. Kanhe, „Performance Comparison of Various Neural Networks for Speech Recognition”, Journal of Physics: Conference Series, t. 2466, doi:10.1088/1742-6596/2466/1/012008 
    55. D. N. Rim, I. Jang, H. Choi, „Deep Neural Networks and End-to-End Learning for Audio Compression”, arXiv, doi:10.48550/arXiv.2105.11681
    56. K. Choi, G. Fazekas, M. Sandler, K. Cho, „Convolutional Recurrent Neural Networks for Music Classification”, arXiv, 2016, doi:10.48550/arXiv.1609.04243
    57. X. Gong, Y. Zhu, H. Zhu, H. Wei, „ChMusinc: A Traditional Chinese Dataset for Evaluation of Instrument Recognition”, Proceedings of the 4th International Conference on Big Data Technologies, 2021, s. 184-189, doi:10.1145/3490322.3490351 
    58. R. Duda, P. Hart, i D. G. Stork, Pattern Classification, t. xx, Wiley Interscience, 2001
    59. J. R. Quinlan, „Induction of decision trees”, Mach Learn 1986, t. 1, s. 81–106, https://doi.org/10.1007/BF00116251 
    60. L. Rokach i O. Maimon, „Decision Trees”, The Data Mining and Knowledge Discovery Handbook, 2005, s. 165–192, doi: 10.1007/0-387-25465-X_9.
    61. T. Cover, P. Hart, "Nearest neighbor pattern classification," IEEE Transactions on Information Theory. 1967, t. 13, nr. 1, s. 21-27, doi: 10.1109/TIT.1967.1053964. 
    62. Z. Zhang, „Introduction to machine learning: k-nearest neighbors”, Annals of Translational Medicine, t. 4, nr 11, s. 218, 2016, doi: 10.21037/atm.2016.03.37 .
    63. C. Cortes, V. Vapnik, „Support-vector networks”, Mach Learn 1995,  t. 20, s. 273–297, https://doi.org/10.1007/BF00994018 
    64. N. Cristianini i J. Shawe-Taylor, An Introduction to Support Vector Machines and Other Kernel-Based Learning Methods, Cambridge University Press, 2000, s. 93–124, doi: 10.1017/CBO9780511801389 .
    65. L. Breiman, „Random forests”, Machine Learning, t. 45, nr 1, s. 5–32, 2001, doi: 10.1023/A:1010933404324.
    66. F. Rosenblatt, „The perceptron: a probabilistic model for information storage and organization in the brain”, Psychological review, t. 65, nr 6, s. 386–408, 1958, doi: 10.1037/h0042519.
    67. I. Goodfellow, Y. Bengio, i A. Courville, Deep Learning, MIT Press, 2016.
    68. R. M. Schmidt, „Recurrent neural networks (RNNs): A gentle introduction and overview”, 2019, doi: 10.48550/arXiv.1912.05911.
    69. A. F. M. Agarap, „Deep Learning using Rectified Linear Units (ReLU)”,  2018, arXiv, doi:10.48550/arXiv.1803.08375
    70. D. M. W. Powers, „Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation”, 2020, doi: 10.48550/arXiv.2010.16061.
    71. T. Fawcett, „Introduction to ROC analysis”, Pattern Recognition Letters, t. 27, s. 861–874, czerwiec 2006, doi: 10.1016/j.patrec.2005.10.010.
    72. K. He, X. Zhang, S. Ren, J. Sun, „Deep Residual Learning for Image Recognition”, 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), s. 770-778, doi:10.1109/CVPR.2016.90
    73. A. Krizhevsky, I. Sutskever, G.E. Hinton, "ImageNet classification with deep convolutional neural networks," Communications of the ACM, 2012, t. 60, s. 84-90, doi: 10.1145/3065386. 
    74. S. Ioffe, C. Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 2015, doi:10.48550/arXiv.1502.03167
    75. J. Thickstun, Z. Harchaoui, S. M. Kakade, „Learning Features of Music from Scratch”, International Conference on Learning Representations (ICLR), 2017, doi:10.48550/arXiv.1611.09827
    76. C. L. Halbert, K. Tretyakov, Biblioteka języka Python, IntervalTree, wersja 3.1.0, dostępna: https://github.com/chaimleib/intervaltree
    77. S. Gururani, M. Sharma, A. Lerch, „An attention mechanism for musical instrument recognition”, 2019, arXiv, doi:10.48550/arXiv.1907.04294
    78. A. Vaswani, N. Shazeer, N. Parmar i in. „Attention Is All You Need”, 2017, arXiv, doi:10.48550/arXiv.1706.03762
    79. F. Yu, V. Koltun, „Multi-Scale Context Aggregation by Dilated Convolutions”, International Conference on Learning Representations, 2016, doi:10.48550/arXiv.1511.07122
    80. C. Szegedy, W. Liu, Y. Jia i in. „Going deeper with convolutions”, 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), doi:10.1109/CVPR.2015.7298594 
    81. J. Hu, L. Shen, S. Albanie i in. „Squeeze-and-Excitation Networks”, 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, doi:10.1109/CVPR.2018.00745 
    82. A. G. Howard, M. Zhu, B. Chen i in. „MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications”, arXiv, 2017, doi:10.48550/arXiv.1704.04861
    83. Python Software Foundation, Dokumentacja Języka Python, wersja 3.8, dostępna https://www.python.org/ 
    84. Anaconda Inc, Dystrybucja oprogramowania Anaconda, dostępna: https://www.anaconda.com/ 
    85. C. R. Harris, K. J. Millman, S. J. van der Walt i. in. „Array programming with NumPy”, 2020, Nature 585, s 357-362, doi:10.1038/s41586-020-2649-2, wersja 1.26.4
    86. The pandas development team, Biblioteka języka Python, Pandas, Zenodo, 2020, wersja 2.2.2, doi:10.5281/zenodo.3509134
    87. B. McFee, M. McVicar, D. Faronbi i in. Biblioteka języka Python, librosa, wersja 0.10.2.post1, doi:10.5281/zenodo.11192913
    88. B. Bechtold, Biblioteka języka Python, soundfile, wersja 0.12.1, dostępna https://github.com/bastibe/python-soundfile
    89. H. Wierstorf, Biblioteka języka Python, audiofile, wersja 1.5.0, dostępna https://audeering.github.io/audiofile/index.html
    90. F. Pedregosa, G. Varoquaux, A. Gramfort i in. „Scikit-learn: Machine Learning in Python”, Journal of Machine Learning Research, 2011, t. 12, s. 2825-2830, dostępna http://scikit-learn.sourceforge.net , wersja 1.5.1
    91. P. Szymański, T. Kajdanowicz, „A scikit-based Python environment for performing multi-label classification”, arXiv, 2017, doi:10.48550/arXiv.1702.01460, wersja 0.2.0
    92. C. O. da Costa-Luis, Biblioteka języka Python, tqdm, wersja 4.66.5, dostępna https://github.com/tqdm/tqdm
    93. M. Abadi, A. Agarwal, P. Barham i in. „TensorFlow: Large-scale machine learning on heterogeneous systems”, 2015, dostępne https://www.tensorflow.org, wersja 2.8.3
    94. F. Chollet, i in. „Keras”, 2015, dostępne https://keras.io, wersja 2.8.0
    95. Weights & Biases Developers, Narzędzie do śledzenia eksperymentów uczenia maszynowego, wersja biblioteki Python 0.17.8, dostępne https://wandb.ai
    96. J. D. Hunter, „Matplotlib: A 2D graphics environment”, Computing in Science & Engineering, 2007, t. 9, nr 3, s. 90-95, doi:10.1109/MCSE.2007.55, wersja 3.5.1
    97. M. L. Waskom, „Seaborn: statistical data visualisation”, The Open Journal, 2021, t. 6, nr 60, s. 3021, doi: 10.21105/joss.03021, wersja 0.13.2
    98. Jetbrains, PyCharm Community, Środowisko programistyczne do języka Python, dostępne https://www.jetbrains.com/pycharm
    99. D. P. Kingma, J. Ba, „Adam: A Method for Stochastic Optimization”, arXiv, 2014, doi:10.48550/arXiv.1412.6980"""


# input_text = """123. Q. Bpple some more words
# 2. D. Canana a few more words
# 45. X. Cat additional text"""

sorted_text = sort_sentences(input_text)
print(sorted_text)
print()

sorted_numbers = sort_by_number(sorted_text)
print(sorted_numbers)

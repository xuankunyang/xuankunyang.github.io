---
title: 🎙️ I have opensourced my VAD project recently
summary: I conducted extensive experiments comparing frame division methods and model performances, with rich visualizations.
date: 2025-05-05
authors:
  - admin
tags:
  - Machine Learning
  - Voice
  - Python
image:
  caption: 'GitHub Repo: [VAD](https://github.com/xuankunyang/Voice-Activity-Detection)'

---

## Voice Activity Detection(VAD)

### Summary

🎯 **Voice Activity Detection (VAD)**, or voice endpoint detection, identifies time segments in an audio signal containing speech. This is a critical preprocessing step for automatic speech recognition (ASR) and voice wake-up systems. This project lays the groundwork for my upcoming ASR project 🤭.

📈 **Workflow Overview**:
The VAD pipeline processes a speech signal as follows:
1. **Preprocessing**: Apply pre-emphasis to enhance high-frequency components.
2. **Framing**: Segment the signal into overlapping frames with frame-level labels.
3. **Windowing**: Apply window functions to mitigate boundary effects.
4. **Feature Extraction**: Extract a comprehensive set of features (e.g., short-time energy, zero-crossing rate, MFCCs, and more).
5. **Binary Classification**: Train models (DNN, Logistic Regression, Linear SVM, GMM) to classify frames as speech or non-speech.
6. **Time-Domain Restoration**: Convert frame-level predictions to time-domain speech segments.

🍻 **Project Highlights**:
I conducted extensive experiments comparing frame division methods (frame length and shift) and model performances, with rich visualizations. For details, see the report in `vad/latex/`. If you're interested in voice technologies, let's connect!

### **View more details in my Blog [VAD](https://xuankunyang.github.io/blogs/vad/)**

# **Happy coding!** 🚀

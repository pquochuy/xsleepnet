import numpy as np
import os

class FilterbankShape(object):

    def lin_tri_filter_shape(self, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        """Compute a linear-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        #lowmel = self.hz2mel(lowfreq)
        #highmel = self.hz2mel(highfreq)
        #melpoints = np.linspace(lowmel,highmel,nfilt+2)
        hzpoints = np.linspace(lowfreq,highfreq,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*hzpoints/samplerate)

        fbank = np.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        fbank = np.transpose(fbank)
        fbank.astype(np.float32)
        return fbank


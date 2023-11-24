 def aoa_by_music(self, input_theta_list=np.arange(-90, 91, 1.), smooth=False, pick_tx=0):
        """
        Computes AoA spectrum by MUSIC.\n
        :param input_theta_list: list of angels, default = -90~90
        :param smooth: whether apply SpotFi smoothing or not, default = False
        :param pick_tx: select 1 tx antenna, default is 0
        :return: AoA spectrum by MUSIC stored in self.data.spectrum
        """
        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        dist_antenna = self.configs.dist_antenna
        torad = self.configs.torad
        subfreq_list = self.configs.subfreq_list
        smoothing = self.commonfunc.smooth_csi
        noise = self.commonfunc.noise_space

        print(self.name, "AoA by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("csi: " + str(self.csi) + "\nPlease load data")

            if smooth is True:
                print(self.name, "apply Smoothing via SpotFi...")

            antenna_list = self.configs.antenna_list
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            spectrum = np.zeros((len(input_theta_list), self.length))

            for i in range(self.length):

                if smooth is True:
                    pass

                noise_space = noise(self.csi[i, :, :, pick_tx])

                if smooth is True:
                    steering_vector = np.exp([-1.j * 2 * np.pi * dist_antenna * (np.sin(theta_list * torad) *
                                              no_antenna).dot(sub_freq) / lightspeed
                                              for no_antenna in antenna_list[:2]
                                              for sub_freq in subfreq_list[:15]])
                else:
                    steering_vector = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                                                antenna_list.T) * center_freq / lightspeed)

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = AoAViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps)
            print(self.name, "AoA by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def aod_by_music(self, input_theta_list=np.arange(-90, 91, 1.), pick_rx=0):
        """
        Computes AoA spectrum by MUSIC.\n
        :param input_theta_list: list of angels, default = -90~90
        :param pick_rx: select 1 tx antenna, default is 0
        :return: AoA spectrum by MUSIC stored in self.data.spectrum
        """
        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        dist_antenna = self.configs.dist_antenna
        torad = self.configs.torad
        noise = self.commonfunc.noise_space

        print(self.name, "AoD by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("csi: " + str(self.csi) + "\nPlease load data")

            antenna_list = self.configs.antenna_list
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            spectrum = np.zeros((len(input_theta_list), self.length))

            for i in range(self.length):

                noise_space = noise(self.csi[i, :, pick_rx, :])

                steering_vector = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                                            antenna_list.T) * center_freq / lightspeed)

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = AoDViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps)
            print(self.name, "AoD by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def tof_by_music(self, input_dt_list=np.arange(-1.e-7, 4.e-7, 1.e-9), pick_tx=0):
        """
        Computes AoA spectrum by MUSIC.\n
        :param input_dt_list: list of tofs, default = -0.5e-7~2e-7
        :param pick_tx: select 1 tx antenna, default is 0
        :return: ToF spectrum by MUSIC stored in self.data.spectrum
        """

        subfreq_list = self.configs.subfreq_list - self.configs.center_freq
        noise = self.commonfunc.noise_space

        print(self.name, "ToF by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("amplitude: " + str(self.csi) + "\nPlease load data")

            dt_list = np.array(input_dt_list[::-1]).reshape(-1, 1)
            spectrum = np.zeros((len(input_dt_list), self.length))

            for i in range(self.length):

                noise_space = noise(self.csi[i, :, :, pick_tx].T)

                steering_vector = np.exp(-1.j * 2 * np.pi * dt_list.dot(subfreq_list.T))

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = ToFViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps)
            print(self.name, "ToF by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def doppler_by_music(self, input_velocity_list=np.arange(-5, 5.01, 0.01),
                         window_length=100,
                         stride=100,
                         pick_rx=0,
                         pick_tx=0,
                         ref_antenna=1,
                         raw_timestamps=False,
                         dynamic=True):
        """
        Computes Doppler spectrum by MUSIC.\n
        Involves self-calibration, windowed dynamic component extraction and resampling (if specified).\n
        :param input_velocity_list: list of velocities. Default = -5~5
        :param window_length: window length for each step
        :param stride: stride for each step
        :param pick_rx: select 1 rx antenna, default is 0. (You can also Specify 'strong' or 'weak')
        :param pick_tx: select 1 tx antenna, default is 0
        :param ref_antenna: select 2 rx antenna for dynamic extraction, default is 1
        :param raw_timestamps: whether to use original timestamps. Default is False
        :param dynamic: whether to use raw CSI or dynamic CSI. Default is True
        :return: Doppler spectrum by MUSIC stored in self.data.spectrum
        """
        sampling_rate = self.configs.sampling_rate
        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        noise = self.commonfunc.noise_space
        wdyn = self.commonfunc.conjmul_dynamic

        print(self.name, "Doppler by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("amplitude: " + str(self.csi) + "\nPlease load data")

            # Each window has (window_length / sampling_rate) seconds of packets
            delay_list = np.arange(0, window_length, 1.).reshape(-1, 1) / sampling_rate
            velocity_list = np.array(input_velocity_list[::-1]).reshape(-1, 1)
            total_strides = (self.length - window_length) // stride

            if pick_rx == 'strong':
                pick_rx = np.argmax(self.show_antenna_strength())
            elif pick_rx == 'weak':
                pick_rx = np.argmin(self.show_antenna_strength())

            spectrum = np.zeros((len(input_velocity_list), total_strides))
            temp_timestamps = np.zeros(total_strides)

            for i in range(total_strides):

                csi_windowed = self.csi[i * stride: i * stride + window_length]

                if dynamic is True:
                    # Using windowed dynamic extraction
                    csi_dynamic = wdyn(csi_windowed, ref='rx', reference_antenna=ref_antenna)
                    noise_space = noise(csi_dynamic[:, :, pick_rx, pick_tx].T)
                else:
                    noise_space = noise(csi_windowed[:, :, pick_rx, pick_tx].T)

                if raw_timestamps is True:
                    # Using original timestamps (possibly uneven intervals)
                    delay_list = self.timestamps[i * stride: i * stride + window_length] - \
                                 self.timestamps[i * stride]

                steering_vector = np.exp(-1.j * 2 * np.pi * center_freq * velocity_list.dot(delay_list.T) / lightspeed)

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

                temp_timestamps[i] = self.timestamps[i * stride]

            self.spectrum = np.log(spectrum)
            self.viewer = DopplerViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps,
                                        xlabels=temp_timestamps)

            print(self.name, "Doppler by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def aoa_tof_by_music(self, input_theta_list=np.arange(-90, 91, 1.),
                         input_dt_list=np.arange(-1.e-7, 2.e-7, 1.e-9),
                         smooth=False):
        """
        Computes AoA-ToF spectrum by MUSIC.\n
        :param input_theta_list:  list of angels, default = -90~90
        :param input_dt_list: list of time measurements, default = 0~8e-8
        :param smooth:  whether apply SpotFi smoothing or not, default = False
        :return:  AoA-ToF spectrum by MUSIC stored in self.data.spectrum
        """

        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        dist_antenna = self.configs.dist_antenna
        torad = self.configs.torad
        subfreq_list = self.configs.subfreq_list
        nsub = self.configs.nsub
        nrx = self.configs.nrx
        smoothing = self.commonfunc.smooth_csi
        noise = self.commonfunc.noise_space

        print(self.name, "AoA-ToF by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("amplitude: " + str(self.csi) + "\nPlease load data")

            if smooth not in (True, False):
                raise ArgError("smooth:" + str(smooth))

            if smooth is True:
                print(self.name, "apply Smoothing via SpotFi...")

            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            dt_list = np.array(input_dt_list).reshape(-1, 1)

            steering_aoa = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                            antenna_list.T) * center_freq / lightspeed).reshape(-1, 1)

            spectrum = np.zeros((self.length, len(input_theta_list), len(input_dt_list)))

            for i in range(self.length):

                if smooth is True:
                    pass

                noise_space = noise(self.csi[i].reshape(1, -1))   # nrx * nsub columns

                for j, tof in enumerate(dt_list):

                    if smooth is True:
                        steering_vector = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                            antenna_list[:2].dot(subfreq_list[:15])) / lightspeed)
                    else:
                        steering_tof = np.exp(-1.j * 2 * np.pi * subfreq_list * tof).reshape(-1, 1)
                        steering_vector = steering_tof.dot(steering_aoa.T).reshape(nsub, len(input_theta_list), nrx)
                        steering_vector = steering_vector.swapaxes(0, 1).reshape(len(input_theta_list), nrx * nsub)

                    a_en = np.conjugate(steering_vector).dot(noise_space)
                    spectrum[i, :, j] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = AoAToFViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps)
            print(self.name, "AoA-ToF by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)
        except ArgError as e:
            print(e, "\nPlease specify smooth=True or False")

    def aoa_doppler_by_music(self, input_theta_list=np.arange(-90, 91, 1.),
                             input_velocity_list=np.arange(-5, 5.05, 0.05),
                             window_length=100,
                             stride=100,
                             raw_timestamps=False,
                             raw_window=False):
        """
        Computes AoA-Doppler spectrum by MUSIC.\n
        :param input_theta_list:  list of angels, default = -90~90
        :param input_velocity_list: list of velocities. Default = -5~5
        :param window_length: window length for each step
        :param stride: stride for each step
        :param raw_timestamps: whether use original timestamps. Default is False
        :param raw_window: whether skip extracting dynamic CSI. Default is False
        :return:  AoA-Doppler spectrum by MUSIC stored in self.data.spectrum
        """

        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        dist_antenna = self.configs.dist_antenna
        sampling_rate = self.configs.sampling_rate
        torad = self.configs.torad
        nrx = self.configs.nrx
        nsub = self.configs.nsub
        noise = self.commonfunc.noise_space
        dynamic = self.commonfunc.conjmul_dynamic

        print(self.name, "AoA-Doppler by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("amplitude: " + str(self.csi) + "\nPlease load data")

            # Each window has ts of packets (1 / sampling_rate * window_length = t)
            delay_list = np.arange(0, window_length, 1.).reshape(-1, 1) / sampling_rate
            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            velocity_list = np.array(input_velocity_list).reshape(-1, 1)

            steering_aoa = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                antenna_list.T) * center_freq / lightspeed).reshape(-1, 1)
            spectrum = np.zeros(((self.length - window_length) // stride, len(input_theta_list),
                                 len(input_velocity_list)))
            temp_timestamps = np.zeros((self.length - window_length) // stride)

            # Using windowed dynamic extraction
            for i in range((self.length - window_length) // stride):

                csi_windowed = self.csi[i * stride: i * stride + window_length]

                if raw_window is True:
                    noise_space = noise(csi_windowed.swapaxes(0, 1).reshape(nsub, window_length * nrx))
                else:
                    # Using windowed dynamic extraction
                    csi_dynamic = dynamic(csi_windowed, ref='rx', reference_antenna=2)
                    noise_space = noise(csi_dynamic.swapaxes(0, 1).reshape(nsub, window_length * nrx))

                if raw_timestamps is True:
                    # Using original timestamps (possibly uneven intervals)
                    delay_list = self.timestamps[i * stride: i * stride + window_length] - \
                                 self.timestamps[i * stride]

                for j, velocity in enumerate(velocity_list):

                    steering_doppler = np.exp(-1.j * 2 * np.pi * center_freq * delay_list * velocity /
                                              lightspeed).reshape(-1, 1)
                    steering_vector = steering_doppler.dot(steering_aoa.T
                                                           ).reshape(len(delay_list), len(input_theta_list), nrx)
                    steering_vector = steering_vector.swapaxes(0, 1
                                                               ).reshape(len(input_theta_list), nrx * len(delay_list))

                    a_en = np.conjugate(steering_vector).dot(noise_space)
                    spectrum[i, :, j] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = AoADopplerViewer(name=self.name, spectrum=self.spectrum, timestamps=temp_timestamps)
            print(self.name, "AoA-Doppler by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)
        except ArgError as e:
            print(e, "\nPlease specify smooth=True or False")


    def sanitize_phase(self):
        """
        Also known as SpotFi Algorithm1.\n
        Removes Sampling Time Offset shared by all rx antennas.\n
        :return: sanitized phase
        """

        nrx = self.configs.nrx
        nsub = self.configs.nsub

        print(self.name, "apply SpotFi Algorithm1 to remove STO...", end='')

        try:
            if self.csi is None:
                raise DataError("phase: " + str(self.csi))

            fit_x = np.concatenate([np.arange(0, nsub) for _ in range(nrx)])
            fit_y = np.unwrap(np.squeeze(self.csi), axis=1).swapaxes(1, 2).reshape(self.length, -1)

            a = np.stack((fit_x, np.ones_like(fit_x)), axis=-1)
            fit = np.linalg.inv(a.T.dot(a)).dot(a.T).dot(fit_y.T).T
            # fit = np.array([np.polyfit(fit_x, fit_y[i], 1) for i in range(self.data.length)])

            phase = np.unwrap(np.angle(self.csi), axis=1) - np.arange(nsub).reshape(
                (1, nsub, 1, 1)) * fit[:, 0].reshape(self.length, 1, 1, 1)
            print("Done")

            self.csi = np.abs(self.csi) * np.exp(1.j * phase)

        except DataError as e:
            print(e, "\nPlease load data")

    def remove_ipo(self, reference_antenna=0, cal_dict=None):
        """
        Calibrates phase with reference csi data files.\n
        Multiple files is supported.\n
        Reference files are recommended to be collected at 50cm at certain degrees (eg. 0, +-30, +-60).\n
        Removes Initial Phase Offset.\n
        :param reference_antenna: select one antenna with which to calculate phase difference between antennas.
        Default is 0
        :param cal_dict: formatted as "{'xx': MyCsi}", where xx is degrees
        :return: calibrated phase
        """
        nrx = self.configs.nrx
        distance_antenna = self.configs.dist_antenna
        torad = self.configs.torad
        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq

        print(self.name, "apply phase calibration according to", str(cal_dict.keys())[10:-1], "...", end='')

        try:
            if self.csi is None:
                raise DataError("csi: " + str(self.csi))

            if reference_antenna not in (0, 1, 2):
                raise ArgError("reference_antenna: " + str(reference_antenna))

            if cal_dict is None:
                raise DataError("reference: " + str(cal_dict))

            ipo = []
            # cal_dict: "{'xx': MyCsi}"

            for key, value in cal_dict.items():

                if not isinstance(value, MyCsi):
                    raise DataError("reference csi: " + str(value) + "\nPlease input MyCsi instance.")

                if value.csi is None:
                    raise DataError("reference phase: " + str(value.csi))

                ref_angle = eval(key)

                ref_csi = value.csi
                ref_diff = np.mean(ref_csi * ref_csi[:, :, reference_antenna][:, :, np.newaxis].conj(),
                                   axis=(0, 1))
                true_diff = np.exp([-1.j * 2 * np.pi * distance_antenna * antenna * center_freq * np.sin(
                    ref_angle * torad) / lightspeed for antenna in range(nrx)]).reshape(-1, 1)

                ipo.append(ref_diff.reshape(-1, 1) * true_diff.conj())

            ipo = np.squeeze(np.mean(ipo, axis=0))

            self.csi = self.csi * ipo[np.newaxis, np.newaxis, :, np.newaxis].conj()

            print("Done")

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e, "\nPlease specify an integer from 0~2")

    def remove_csd(self, HT=False):
        """
        Remove CSD based on values in 802.11 standard.\n
        Requires 3 tx.\n
        non-HT: -200ns, -100ns\n
        HT: -400ns, -200ns\n
        :param HT: Default is False
        :return: CSI with CSD removed
        """

        print(self.name, "removing CSD...", end='')

        try:
            if self.csi is None:
                raise DataError("csi: " + str(self.csi))

            if self.configs.ntx != 3:
                raise DataError(str(self.csi) + 'does not have multiple tx')
            else:
                if HT:
                    csd_1 = np.exp(2.j * np.pi * self.configs.subfreq_list * (-400) * 1.e-9)
                    csd_2 = np.exp(2.j * np.pi * self.configs.subfreq_list * (-200) * 1.e-9)
                else:
                    csd_1 = np.exp(2.j * np.pi * self.configs.subfreq_list * (-200) * 1.e-9)
                    csd_2 = np.exp(2.j * np.pi * self.configs.subfreq_list * (-100) * 1.e-9)

            self.csi[:, :, :, 1] = self.csi[:, :, :, 1] * csd_1
            self.csi[:, :, :, 2] = self.csi[:, :, :, 2] * csd_2

            print("Done")

        except DataError as e:
            print(e, "\nPlease load data")

    def show_csd(self):
        if self.configs.ntx != 3:
            return
        else:
            csd_1 = self.csi[..., 0] * self.csi[..., 1].conj()
            csd_2 = self.csi[..., 0] * self.csi[..., 2].conj()

            csd_1 = np.unwrap(np.squeeze(np.angle(np.mean(csd_1, axis=0)))) / (2 * np.pi * self.configs.subfreq_list) * 1.e9
            csd_2 = np.unwrap(np.squeeze(np.angle(np.mean(csd_2, axis=0)))) / (2 * np.pi * self.configs.subfreq_list) * 1.e9

            plt.subplot(2, 1, 1)

            for rx in range(self.configs.nrx):
                plt.plot(csd_1[:, rx], label='rx'+str(rx))
            plt.xlabel('Sub')
            plt.ylabel('CSD/ns')
            plt.title('CSD_1')
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            for rx in range(self.configs.nrx):
                plt.plot(csd_2[:, rx], label='rx' + str(rx))
            plt.xlabel('Sub')
            plt.ylabel('CSD/ns')
            plt.title('CSD_2')
            plt.legend()
            plt.grid()

            plt.suptitle('CSD')
            plt.tight_layout()
            plt.show()

    def extract_dynamic(self, mode='overall-multiply',
                        ref='rx',
                        ref_antenna=0,
                        window_length=100,
                        stride=100,
                        subtract_mean=False,
                        **kwargs):
        """
        Removes the static component from csi.\n
        :param mode: 'overall' or 'running' (in terms of averaging) or 'highpass'. Default is 'overall'
        :param ref: 'rx' or 'tx'
        :param window_length: if mode is 'running', specify a window length for running mean. Default is 100
        :param stride: if mode is 'running', specify a stride for running mean. Default is 100
        :param ref_antenna: select one antenna with which to remove random phase offsets. Default is 0
        :param subtract_mean: whether to subtract mean of cSI. Default is False
        :return: dynamic component of csi
        """
        nrx = self.configs.nrx
        nsub = self.configs.nsub
        ntx = self.configs.ntx
        dynamic = self.commonfunc.conjmul_dynamic
        division = self.commonfunc.divison_dynamic
        highpass = self.commonfunc.highpass

        print(self.name, "apply dynamic component extraction: " + mode + " versus " + ref + str(ref_antenna) + "...", end='')

        try:
            if self.csi is None:
                raise DataError("csi data")

            if ref_antenna not in range(nrx):
                raise ArgError("reference_antenna: " + str(ref_antenna) + "\nPlease specify an integer from 0~2")

            if ref_antenna is None:
                strengths = self.show_antenna_strength()
                ref_antenna = np.argmax(strengths)

            if mode == 'overall-multiply':
                dynamic_csi = dynamic(self.csi, ref, ref_antenna, subtract_mean)

            elif mode == 'overall-divide':
                dynamic_csi = division(self.csi, ref, ref_antenna, subtract_mean)

            elif mode == 'running-multiply':
                dynamic_csi = np.zeros((self.length, nsub, nrx, ntx), dtype=complex)
                for step in range((self.length - window_length) // stride):
                    dynamic_csi[step * stride: step * stride + window_length] = dynamic(
                        self.csi[step * stride: step * stride + window_length], ref, ref_antenna, subtract_mean)

            elif mode == 'running-divide':
                dynamic_csi = np.zeros((self.length, nsub, nrx, ntx), dtype=complex)
                for step in range((self.length - window_length) // stride):
                    dynamic_csi[step * stride: step * stride + window_length] = division(
                        self.csi[step * stride: step * stride + window_length], ref, ref_antenna, subtract_mean)

            elif mode == 'highpass':
                b, a = highpass(**kwargs)
                dynamic_csi = np.zeros_like(self.csi)
                for sub in range(nsub):
                    for rx in range(nrx):
                        for tx in range(ntx):
                            dynamic_csi[:, sub, rx, tx] = signal.filtfilt(b, a, self.csi[:, sub, rx, tx])

            else:
                raise ArgError("mode: " + str(mode) +
                               "\nPlease specify mode=\"overall-multiply\", \"overall-divide\", \"running-divide\"or "
                               "\"highpass\"")

            self.csi = dynamic_csi
            print("Done")

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e)

    def resample_packets(self, sampling_rate=100):
        """
        Resample from raw CSI to reach a specified sampling rate.\n
        Strongly recommended when uniform interval is required.\n
        :param sampling_rate: sampling rate in Hz after resampling. Must be less than 3965.
        Default is 100
        :return: Resampled csi data
        """
        print(self.name, "resampling at " + str(sampling_rate) + "Hz...", end='')

        try:
            if self.csi is None:
                raise DataError("csi data")

            if not isinstance(sampling_rate, int) or sampling_rate >= self.actual_sr:
                raise ArgError("sampling_rate: " + str(sampling_rate))

            new_interval = 1. / sampling_rate

            new_length = int(self.timestamps[-1] * sampling_rate) + 1  # Flooring
            resample_indicies = []

            for i in range(new_length):

                index = np.searchsorted(self.timestamps, i * new_interval)

                if index > 0 and (
                        index == self.length or
                        abs(self.timestamps[index] - i * new_interval) >
                        abs(self.timestamps[index - 1] - i * new_interval)):
                    index -= 1

                resample_indicies.append(index)

            self.csi = self.csi[resample_indicies]
            self.timestamps = self.timestamps[resample_indicies]
            self.length = new_length
            self.actual_sr = sampling_rate

            print("Done")

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e, "\nPlease specify an integer less than the current sampling rate")


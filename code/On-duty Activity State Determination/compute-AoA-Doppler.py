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

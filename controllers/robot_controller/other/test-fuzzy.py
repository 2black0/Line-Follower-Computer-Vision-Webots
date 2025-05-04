self.Error = ctrl.Antecedent(np.arange(-80, 80.1, 0.1), 'Error')
self.DerivativeError = ctrl.Antecedent(np.arange(-20, 20.1, 0.1), 'DerivativeError')
self.DeltaSpeed = ctrl.Consequent(np.arange(-2.0, 2.01, 0.01), 'DeltaSpeed')

self.Error.automf(5, names=['NB', 'NM', 'ZE', 'PM', 'PB'])
self.DerivativeError.automf(5, names=['NB', 'NM', 'ZE', 'PM', 'PB'])
self.DeltaSpeed.automf(5, names=['KB', 'KS', 'ST', 'TS', 'TB'])

self.Rules = [
    ctrl.Rule(self.Error['NB'] & self.DerivativeError['NB'], self.DeltaSpeed['KB']),
    ctrl.Rule(self.Error['NB'] & self.DerivativeError['NM'], self.DeltaSpeed['KB']),
    ctrl.Rule(self.Error['NB'] & self.DerivativeError['ZE'], self.DeltaSpeed['KS']),
    ctrl.Rule(self.Error['NB'] & self.DerivativeError['PM'], self.DeltaSpeed['KS']),
    ctrl.Rule(self.Error['NB'] & self.DerivativeError['PB'], self.DeltaSpeed['ST']),

    ctrl.Rule(self.Error['NM'] & self.DerivativeError['NB'], self.DeltaSpeed['KB']),
    ctrl.Rule(self.Error['NM'] & self.DerivativeError['NM'], self.DeltaSpeed['KS']),
    ctrl.Rule(self.Error['NM'] & self.DerivativeError['ZE'], self.DeltaSpeed['KS']),
    ctrl.Rule(self.Error['NM'] & self.DerivativeError['PM'], self.DeltaSpeed['ST']),
    ctrl.Rule(self.Error['NM'] & self.DerivativeError['PB'], self.DeltaSpeed['ST']),

    ctrl.Rule(self.Error['ZE'] & self.DerivativeError['NB'], self.DeltaSpeed['KS']),
    ctrl.Rule(self.Error['ZE'] & self.DerivativeError['NM'], self.DeltaSpeed['ST']),
    ctrl.Rule(self.Error['ZE'] & self.DerivativeError['ZE'], self.DeltaSpeed['ST']),
    ctrl.Rule(self.Error['ZE'] & self.DerivativeError['PM'], self.DeltaSpeed['TS']),
    ctrl.Rule(self.Error['ZE'] & self.DerivativeError['PB'], self.DeltaSpeed['TS']),

    ctrl.Rule(self.Error['PM'] & self.DerivativeError['NB'], self.DeltaSpeed['ST']),
    ctrl.Rule(self.Error['PM'] & self.DerivativeError['NM'], self.DeltaSpeed['ST']),
    ctrl.Rule(self.Error['PM'] & self.DerivativeError['ZE'], self.DeltaSpeed['TS']),
    ctrl.Rule(self.Error['PM'] & self.DerivativeError['PM'], self.DeltaSpeed['TS']),
    ctrl.Rule(self.Error['PM'] & self.DerivativeError['PB'], self.DeltaSpeed['TB']),

    ctrl.Rule(self.Error['PB'] & self.DerivativeError['NB'], self.DeltaSpeed['ST']),
    ctrl.Rule(self.Error['PB'] & self.DerivativeError['NM'], self.DeltaSpeed['TS']),
    ctrl.Rule(self.Error['PB'] & self.DerivativeError['ZE'], self.DeltaSpeed['TB']),
    ctrl.Rule(self.Error['PB'] & self.DerivativeError['PM'], self.DeltaSpeed['TB']),
    ctrl.Rule(self.Error['PB'] & self.DerivativeError['PB'], self.DeltaSpeed['TB'])
]

self.DeltaSpeedControl = ctrl.ControlSystem(self.Rules)
self.DeltaSpeedSim = ctrl.ControlSystemSimulation(self.DeltaSpeedControl)


ErrorFollow = max(min(ErrorFollow, 80), -80)
DerivativeFollow = ErrorFollow - self.PreviousErrorFollow
DerivativeFollow = max(min(DerivativeFollow, 20), -20)
self.DeltaSpeedSim.input['Error'] = ErrorFollow
self.DeltaSpeedSim.input['DerivativeError'] = DerivativeFollow
self.DeltaSpeedSim.compute()
self.PreviousErrorFollow = ErrorFollow
return self.DeltaSpeedSim.output['DeltaSpeed']
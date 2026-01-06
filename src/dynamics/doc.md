Total force acting,\
$F_{yf} + F_{yr} = m(\ddot{y} + \dot{\psi}\dot{x})$

$\dot{\psi}$ - Stearing speed

Moment,\
$I\ddot{\psi} = l_f F_{yf} - l_r F_{yr}$

Force acting on front and rear wheel are,\
$F_{yf} = 2C_{\alpha f} \alpha_f$\
$F_{yr} = 2C_{\alpha r} \alpha_r$


$\alpha_f$ - Slip angle of frontal wheel\
$\alpha_r$ - Slip angle of rear wheel

$F_{yf} = 2C_{\alpha f} (\delta - \theta_{vf})$\
$F_{yr} = 2C_{\alpha r} (-\theta_{vr})$

$C_{\alpha f}$ - Cornering stiffness of fornt wheel\
$C_{\alpha r}$ - Cornering stiffness of fornt wheel


$\theta_{vf}$ - Angle between velocity vector and axis of the vehicle for frontal wheel\
$\theta_{vr}$ - Angle between velocity vector and axis of the vehicle for rear wheel\
$\theta_{vf} = \tan^{-1}(\frac{\dot{y}+l_f\dot{\psi}}{\dot{x}})$\
$\theta_{vr} = \tan^{-1}(\frac{\dot{y}-l_r\dot{\psi}}{\dot{x}})$

Since $\theta$ is small,\
$\theta_{vf} = \frac{\dot{y}+l_f\dot{\psi}}{\dot{x}}$\
$\theta_{vr} = \frac{\dot{y}-l_r\dot{\psi}}{\dot{x}}$

$l\dot{\psi}$ - Due to rotation

$F_{yf} + F_{yr} = m(\ddot{y} + \dot{\psi}\dot{x})$

$m(\ddot{y} + \dot{\psi}\dot{x}) =2C_{\alpha f} (\delta - \theta_{vf}) +  2C_{\alpha r} (-\theta_{vr}) $

$m(\ddot{y} + \dot{\psi}\dot{x}) =2C_{\alpha f} (\delta - \frac{\dot{y}+l_f\dot{\psi}}{\dot{x}}) +  2C_{\alpha r} (-\frac{\dot{y}-l_r\dot{\psi}}{\dot{x}}) $

$m(\ddot{y} + \dot{\psi}\dot{x}) =2C_{\alpha f} (\delta - \frac{\dot{y}+l_f\dot{\psi}}{\dot{x}}) +  2C_{\alpha r} (-\frac{\dot{y}-l_r\dot{\psi}}{\dot{x}}) $

$m\ddot{y} = -m\dot{\psi}\dot{x} - 2C_{\alpha f} ( \frac{\dot{y}+l_f\dot{\psi}}{\dot{x}}) - 2C_{\alpha r} (\frac{\dot{y}-l_r\dot{\psi}}{\dot{x}}) +2C_{\alpha f}\delta$


### State-Space Form
$\begin{bmatrix} \ddot{y} \\ \ddot{\psi} \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} + \begin{bmatrix} \dot{y} \\ \dot{\psi} \end{bmatrix} \delta$

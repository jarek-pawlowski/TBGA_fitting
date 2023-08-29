
    subroutine H_k_MoSe2WSe2(kx,ky,Ed_up,Ep1_up,Ep0_up,Vdp_sigma_up,Vdp_pi_up,Vdd_sigma_up,Vdd_pi_up,Vdd_delta_up,Vpp_sigma_up,Vpp_pi_up, &
    Ep1_odd_up,Ep0_odd_up,Ed_odd_up,lambda_M_up,lambda_X2_up,Ed_down,Ep1_down,Ep0_down,Vdp_sigma_down,Vdp_pi_down,Vdd_sigma_down,Vdd_pi_down,&
    Vdd_delta_down,Vpp_sigma_down,Vpp_pi_down,Ep1_odd_down,Ep0_odd_down,Ed_odd_down,lambda_M_down,lambda_X2_down,Vpp_sigma_inter,Vpp_pi_inter,&
    Vdd_sigma_inter,Vdd_pi_inter,Vdd_delta_inter,H_k)           
		implicit none
            
        double precision, intent(in) :: kx, ky, &
                                        Ed_up, Ep1_up, Ep0_up, Vdp_sigma_up, Vdp_pi_up, Vdd_sigma_up, Vdd_pi_up, Vdd_delta_up, Vpp_sigma_up, &
                                        Vpp_pi_up, Ep1_odd_up, Ep0_odd_up, Ed_odd_up, lambda_M_up, lambda_X2_up, &
                                        Ed_down, Ep1_down, Ep0_down, Vdp_sigma_down, Vdp_pi_down, Vdd_sigma_down, Vdd_pi_down, Vdd_delta_down, &
                                        Vpp_sigma_down, Vpp_pi_down, Ep1_odd_down, Ep0_odd_down, Ed_odd_down, lambda_M_down, lambda_X2_down, &
                                        Vpp_sigma_inter, Vpp_pi_inter, Vdd_sigma_inter, Vdd_pi_inter, Vdd_delta_inter
		double complex, intent(out)  :: H_k(44,44)
		double precision, parameter  :: pi = 4.0d0*atan(1.0d0)
        double precision             :: lattice_const, d1, d2_up, d2_down, d_up, d_down, layer_dist, dz_pp, dz_dd, d_pp, d_dd, &
                                        R1x_pp, R1y_pp, R2x_pp, R2y_pp, R3x_pp, R3y_pp, &
                                        R1x_dd, R1y_dd, R2x_dd, R2y_dd, R3x_dd, R3y_dd
        double complex               :: g0, g2, g4, f_m1_up, f_0_up, f_p1_up, f_m1_down, f_0_down, f_p1_down, &
                                        V1_up, V2_up, V3_up, V4_up, V5_up, V6_up, V7_up, V8_up, &
                                        W1_up, W2_up, W3_up, W4_up, W5_up, W6_up, W7_up, W8_up, W9_up, &
                                        V1_down, V2_down, V3_down, V4_down, V5_down, V6_down, V7_down, V8_down, &
                                        W1_down, W2_down, W3_down, W4_down, W5_down, W6_down, W7_down, W8_down, W9_down, &
                                        W10, W11, W12, W13, W14, W15, W16, h1_pp, h1_dd, h2_dd, h3_dd
        
  
        ! ------- Geometry Parameters (lattice constanst for MoSe2) -------
        lattice_const = 3.323d0 
        d1            = lattice_const/sqrt(3.0d0) 
        d2_up         = 1.669d0
        d2_down       = 1.680d0
        d_up          = sqrt(d1**2.0d0+d2_up**2.0d0) 
        d_down        = sqrt(d1**2.0d0+d2_down**2.0d0) 
        layer_dist    = 6.4d0 
        dz_pp         = layer_dist - d2_up - d2_down
        d_pp          = sqrt( (lattice_const**2.0d0 / 3.0d0) + (dz_pp**2.0d0) ) 
        dz_dd         = layer_dist
        d_dd          = sqrt( (lattice_const**2.0d0 / 3.0d0) + (dz_dd**2.0d0) ) 
        R1x_pp        = -d1/2.0d0 
        R1y_pp        = d1*sqrt(3.0d0)/2.0d0 
        R2x_pp        = -d1/2.0d0  
        R2y_pp        = -d1*sqrt(3.0d0)/2.0d0 
        R3x_pp        = d1
        R3y_pp        = 0.0d0 
        R1x_dd        = d1/2.0d0 
        R1y_dd        = d1*sqrt(3.0d0)/2.0d0 
        R2x_dd        = d1/2.0d0  
        R2y_dd        = -d1*sqrt(3.0d0)/2.0d0 
        R3x_dd        = -d1
        R3y_dd        = 0.0d0 

        
        ! --- Constructing the Hamiltonian ---

            g0        =  4.0d0*cos(3.0d0/2.0d0*kx*d1)*cos(sqrt(3.0d0)/2.0d0*ky*d1) + &
                         2.0d0*cos(sqrt(3.0d0)*ky*d1) 
            g2        =  2.0d0*cos(3.0d0/2.0d0*kx*d1+sqrt(3.0d0)/2.0d0*ky*d1)*exp(dcmplx(0.0d0,1.0d0)*pi/3.0d0)  + &
                         2.0d0*cos(3.0d0/2.0d0*kx*d1-sqrt(3.0d0)/2.0d0*ky*d1)*exp(-dcmplx(0.0d0,1.0d0)*pi/3.0d0) - &
                         2.0d0*cos(sqrt(3.0d0)*ky*d1) 
            g4        =  2.0d0*cos(3.0d0/2.0d0*kx*d1+sqrt(3.0d0)/2.0d0*ky*d1)*exp(dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0)  + &
                         2.0d0*cos(3.0d0/2.0d0*kx*d1-sqrt(3.0d0)/2.0d0*ky*d1)*exp(-dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0) + &
                         2.0d0*cos(sqrt(3.0d0)*ky*d1) 
            f_m1_up   = exp(-dcmplx(0.0d0,1.0d0)*kx*d1) + &
                        exp(dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0) + &
                        exp(dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0)*exp(dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0) 
            f_0_up    = exp(-dcmplx(0.0d0,1.0d0)*kx*d1) + &
                        exp(dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0)*exp(dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0) + &
                        exp(dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0) 
            f_p1_up   = exp(-dcmplx(0.0d0,1.0d0)*kx*d1) + &
                        exp(dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0) + &
                        exp(dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0) 
            f_m1_down = exp(dcmplx(0.0d0,1.0d0)*kx*d1) + &
                        exp(-dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0)*exp(dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0) + &
                        exp(-dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0) 
            f_0_down  = exp(dcmplx(0.0d0,1.0d0)*kx*d1) + &
                        exp(-dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0) + &
                        exp(-dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0)*exp( dcmplx(0.0d0,1.0d0)*2.0d0*pi/3.0d0) 
            f_p1_down  = exp(dcmplx(0.0d0,1.0d0)*kx*d1) + &
                        exp(-dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0) + &
                        exp(-dcmplx(0.0d0,1.0d0)*kx*d1/2.0d0)*exp(-dcmplx(0.0d0,1.0d0)*sqrt(3.0d0)*ky*d1/2.0d0) 

            ! layer up
            V1_up  =  1.0d0/sqrt(2.0d0)*d1/d_up*( sqrt(3.0d0)/2.0d0*Vdp_sigma_up*((d2_up/d_up)**(2.0d0)-1) - Vdp_pi_up*((d2_up/d_up)**(2.0d0)+1) ) 
            V2_up  =  0.5d0*( sqrt(3.0d0)*Vdp_sigma_up-2.0d0*Vdp_pi_up )*(d2_up/d_up)*(d1/d_up)**(2.0d0) 
            V3_up  =  1.0d0/sqrt(2.0d0)*( sqrt(3.0d0)/2.0d0*Vdp_sigma_up-Vdp_pi_up)*(d1/d_up)**(3.0d0) 
            V4_up  =  0.5d0*( (3.0d0*(d2_up/d_up)**(2.0d0)-1)*Vdp_sigma_up - (2.0d0*sqrt(3.0d0)*(d2_up/d_up)**(2.0d0))*Vdp_pi_up )*(d1/d_up) 
            V5_up  =  1.0d0/sqrt(2.0d0)*(d2_up/d_up)*( (3.0d0*(d2_up/d_up)**(2.0d0)-1)*Vdp_sigma_up - (2.0d0*sqrt(3.0d0)*((d2_up/d_up)**(2.0d0)-1))*Vdp_pi_up ) 
            V6_up  =  1.0d0/sqrt(2.0d0)*(d2_up/d_up)*( ((d1/d_up)**2.0d0)*(sqrt(3.0d0)*Vdp_sigma_up-2.0d0*Vdp_pi_up)+2.0d0*Vdp_pi_up ) 
            V7_up  =  1.0d0/sqrt(2.0d0)*(d2_up*d1**2.0d0)/d_up**3.0d0 * ( sqrt(3.0d0)*Vdp_sigma_up- 2.0d0*Vdp_pi_up ) 
            V8_up  =  d1/d_up * ( ((d2_up/d_up)**2.0d0)*(sqrt(3.0d0)*Vdp_sigma_up-2.0d0*Vdp_pi_up)+Vdp_pi_up ) 
            W1_up  =  0.125d0*(3.0d0*Vdd_sigma_up+4*Vdd_pi_up+Vdd_delta_up) 
            W2_up  =  0.25d0*(Vdd_sigma_up+3.0d0*Vdd_delta_up) 
            W3_up  = -sqrt(3.0d0)/(4*sqrt(2.0d0))*(Vdd_sigma_up-Vdd_delta_up) 
            W4_up  =  0.125d0*(3.0d0*Vdd_sigma_up-4*Vdd_pi_up+Vdd_delta_up) 
            W5_up  =  0.5d0*(Vpp_sigma_up+Vpp_pi_up) 
            W6_up  =  Vpp_pi_up 
            W7_up  =  0.5d0*(Vpp_sigma_up-Vpp_pi_up) 
            W8_up  =  0.5d0*(Vdd_pi_up + Vdd_delta_up) 
            W9_up  =  0.5d0*(Vdd_pi_up - Vdd_delta_up) 
    
            ! layer down
            V1_down  =  1.0d0/sqrt(2.0d0)*d1/d_down*( sqrt(3.0d0)/2.0d0*Vdp_sigma_down*((d2_down/d_down)**(2.0d0)-1) - Vdp_pi_down*((d2_down/d_down)**(2.0d0)+1) ) 
            V2_down  =  0.5d0*( sqrt(3.0d0)*Vdp_sigma_down-2.0d0*Vdp_pi_down )*(d2_down/d_down)*(d1/d_down)**(2.0d0) 
            V3_down  =  1.0d0/sqrt(2.0d0)*( sqrt(3.0d0)/2.0d0*Vdp_sigma_down-Vdp_pi_down)*(d1/d_down)**(3.0d0) 
            V4_down  =  0.5d0*( (3.0d0*(d2_down/d_down)**(2.0d0)-1)*Vdp_sigma_down - (2.0d0*sqrt(3.0d0)*(d2_down/d_down)**(2.0d0))*Vdp_pi_down )*(d1/d_down) 
            V5_down  =  1.0d0/sqrt(2.0d0)*(d2_down/d_down)*( (3.0d0*(d2_down/d_down)**(2.0d0)-1)*Vdp_sigma_down - (2.0d0*sqrt(3.0d0)*((d2_down/d_down)**(2.0d0)-1))*Vdp_pi_down ) 
            V6_down  =  1.0d0/sqrt(2.0d0)*(d2_down/d_down)*( ((d1/d_down)**2.0d0)*(sqrt(3.0d0)*Vdp_sigma_down-2.0d0*Vdp_pi_down)+2.0d0*Vdp_pi_down ) 
            V7_down  =  1.0d0/sqrt(2.0d0)*(d2_down*d1**2.0d0)/d_down**3.0d0 * ( sqrt(3.0d0)*Vdp_sigma_down- 2.0d0*Vdp_pi_down ) 
            V8_down  =  d1/d_down * ( ((d2_down/d_down)**2.0d0)*(sqrt(3.0d0)*Vdp_sigma_down-2.0d0*Vdp_pi_down)+Vdp_pi_down ) 
            W1_down  =  1.0d0/8*(3.0d0*Vdd_sigma_down+4*Vdd_pi_down+Vdd_delta_down) 
            W2_down  =  1.0d0/4*(Vdd_sigma_down+3.0d0*Vdd_delta_down) 
            W3_down  = -sqrt(3.0d0)/(4*sqrt(2.0d0))*(Vdd_sigma_down-Vdd_delta_down) 
            W4_down  =  1.0d0/8*(3.0d0*Vdd_sigma_down-4*Vdd_pi_down+Vdd_delta_down) 
            W5_down  =  0.5d0*(Vpp_sigma_down+Vpp_pi_down) 
            W6_down  =  Vpp_pi_down 
            W7_down  =  0.5d0*(Vpp_sigma_down-Vpp_pi_down) 
            W8_down  =  0.5d0*(Vdd_pi_down + Vdd_delta_down) 
            W9_down  =  0.5d0*(Vdd_pi_down - Vdd_delta_down) 

            ! interlayer interactions
            W12    = ((dz_pp/d_pp)**2.0d0)*Vpp_sigma_inter + (1.0d0-(dz_pp/d_pp)**2.0d0)*Vpp_pi_inter 
            W13    = Vdd_sigma_inter * (( 0.5d0*((d1/d_dd)**2.0d0) - ((dz_dd/d_dd)**2.0d0) )**2.0d0)   + &
                     Vdd_pi_inter    * ( 3.0d0*((dz_dd/d_dd)**2.0d0)*(1.0d0 - ((dz_dd/d_dd)**2.0d0)) ) + &
                     Vdd_delta_inter * ( 0.75d0 * (((dz_dd/d_dd)**2.0d0)**2.0d0) )
            W14    = ((d1/d_dd)**2.0d0) * &
                     ( 0.75d0*Vdd_sigma_inter + 0.25d0*((dz_dd/d_dd)**2.0d0)*Vdd_delta_inter + 0.5d0*((dz_dd/d_dd)**2.0d0)*Vdd_pi_inter )
            W15    = 2.0d0*((d1/d_dd)**2.0d0)*Vdd_pi_inter + ((dz_dd/d_dd)**2.0d0)*Vdd_delta_inter
            W16    = (3.0d0/16.0d0)*((d1/d_dd)**4.0d0) * &
                     ( 3.0d0*Vdd_sigma_inter + Vdd_delta_inter - 4.0d0*Vdd_pi_inter )
            
            h1_pp  = exp(dcmplx(0.0d0,1.0d0)*kx*R1x_pp)*exp(dcmplx(0.0d0,1.0d0)*ky*R1y_pp) + &
                     exp(dcmplx(0.0d0,1.0d0)*kx*R2x_pp)*exp(dcmplx(0.0d0,1.0d0)*ky*R2y_pp) + &
                     exp(dcmplx(0.0d0,1.0d0)*kx*R3x_pp)*exp(dcmplx(0.0d0,1.0d0)*ky*R3y_pp) 
            h1_dd  = exp(dcmplx(0.0d0,1.0d0)*kx*R1x_dd)*exp(dcmplx(0.0d0,1.0d0)*ky*R1y_dd) + &
                     exp(dcmplx(0.0d0,1.0d0)*kx*R2x_dd)*exp(dcmplx(0.0d0,1.0d0)*ky*R2y_dd) + &
                     exp(dcmplx(0.0d0,1.0d0)*kx*R3x_dd)*exp(dcmplx(0.0d0,1.0d0)*ky*R3y_dd) 
            h2_dd  = (-0.5d0)*exp(dcmplx(0.0d0,1.0d0)*kx*R1x_dd)*exp(dcmplx(0.0d0,1.0d0)*ky*R1y_dd) + &
                     (-0.5d0)*exp(dcmplx(0.0d0,1.0d0)*kx*R2x_dd)*exp(dcmplx(0.0d0,1.0d0)*ky*R2y_dd) + &
                              exp(dcmplx(0.0d0,1.0d0)*kx*R3x_dd)*exp(dcmplx(0.0d0,1.0d0)*ky*R3y_dd) 
            h3_dd  = (dcmplx(1.0d0,0.0d0)+dcmplx(0.0d0,2.0d0*sqrt(3.0d0)/3.0d0))*exp(dcmplx(0.0d0,1.0d0)*kx*R1x_dd)*exp(dcmplx(0.0d0,1.0d0)*ky*R1y_dd) + &
                     (dcmplx(1.0d0,0.0d0)-dcmplx(0.0d0,2.0d0*sqrt(3.0d0)/3.0d0))*exp(dcmplx(0.0d0,1.0d0)*kx*R2x_dd)*exp(dcmplx(0.0d0,1.0d0)*ky*R2y_dd)
	            
    
            H_k = 0.0d0
            ! diagonal part
            H_k(1,1)   = Ed_up  + (W1_up*g0)
            H_k(2,2)   = Ed_up  + (W2_up*g0)
            H_k(3,3)   = Ed_up  + (W1_up*g0) 
            H_k(4,4)   = Ep1_up + (W5_up*g0)
            H_k(5,5)   = Ep0_up + (W6_up*g0)
            H_k(6,6)   = Ep1_up + (W5_up*g0)
            H_k(7,7)   = Ed_odd_up  + (W8_up*g0)
            H_k(8,8)   = Ed_odd_up  + (W8_up*g0) 
            H_k(9,9)   = Ep1_odd_up + (W5_up*g0) 
            H_k(10,10) = Ep0_odd_up + (W6_up*g0)
            H_k(11,11) = Ep1_odd_up + (W5_up*g0)
            H_k(12,12) = Ed_down  + (W1_down*g0)
            H_k(13,13) = Ed_down  + (W2_down*g0)
            H_k(14,14) = Ed_down  + (W1_down*g0)
            H_k(15,15) = Ep1_down + (W5_down*g0)
            H_k(16,16) = Ep0_down + (W6_down*g0)
            H_k(17,17) = Ep1_down + (W5_down*g0) 
            H_k(18,18) = Ed_odd_down  + (W8_down*g0)
            H_k(19,19) = Ed_odd_down  + (W8_down*g0)
            H_k(20,20) = Ep1_odd_down + (W5_down*g0)
            H_k(21,21) = Ep0_odd_down + (W6_down*g0)
            H_k(22,22) = Ep1_odd_down + (W5_down*g0)
            ! off-diagonal part W-dependent
            H_k(1,2)   =  W3_up*g2 
            H_k(1,3)   =  W4_up*g4 
            H_k(2,3)   =  W3_up*g2 
            H_k(4,6)   = -W7_up*g2 
            H_k(7,8)   = -W9_up*g2 
            H_k(9,11)  = -W7_up*g2 
            H_k(12,13) =  W3_down*g2 
            H_k(12,14) =  W4_down*g4 
            H_k(13,14) =  W3_down*g2 
            H_k(15,17) = -W7_down*g2 
            H_k(18,19) = -W9_down*g2 
            H_k(20,22) = -W7_down*g2  
            ! off-diagonal part V-dependent
            H_k(1,4)   =  V1_up*f_m1_up 
            H_k(1,5)   = -V2_up*f_0_up 
            H_k(1,6)   =  V3_up*f_p1_up 
            H_k(2,4)   = -V4_up*f_0_up
            H_k(2,5)   = -V5_up*f_p1_up 
            H_k(2,6)   =  V4_up*f_m1_up 
            H_k(3,4)   = -V3_up*f_p1_up 
            H_k(3,5)   = -V2_up*f_m1_up 
            H_k(3,6)   = -V1_up*f_0_up 
            H_k(7,9)   = -V6_up*f_p1_up 
            H_k(7,10)  = -V8_up*f_m1_up 
            H_k(7,11)  =  V7_up*f_0_up 
            H_k(8,9)   =  V7_up*f_m1_up 
            H_k(8,10)  =  V8_up*f_0_up 
            H_k(8,11)  = -V6_up*f_p1_up 
            H_k(12,15) =  V1_down*f_m1_down 
            H_k(12,16) = -V2_down*f_0_down
            H_k(12,17) =  V3_down*f_p1_down 
            H_k(13,15) = -V4_down*f_0_down
            H_k(13,16) = -V5_down*f_p1_down 
            H_k(13,17) =  V4_down*f_m1_down 
            H_k(14,15) = -V3_down*f_p1_down 
            H_k(14,16) = -V2_down*f_m1_down 
            H_k(14,17) = -V1_down*f_0_down
            H_k(18,20) = -V6_down*f_p1_down 
            H_k(18,21) = -V8_down*f_m1_down 
            H_k(18,22) =  V7_down*f_0_down
            H_k(19,20) =  V7_down*f_m1_down 
            H_k(19,21) =  V8_down*f_0_down
            H_k(19,22) = -V6_down*f_p1_down 
            ! layer interactions
            H_k(2,13)  = -0.5d0*W13*h1_dd
            H_k(5,16)  = -0.5d0*W12*h1_pp
            H_k(1,14) = h2_dd*W14 + h1_dd*W15 + h3_dd*W16
            H_k(3,12) = conjg(H_k(1,14))
            
            ! SOC
            H_k(23:44,23:44) = H_k(1:22,1:22)
            ! spin1
            ! even_up
            H_k(1,1)               = H_k(1,1)               + (-1.0d0)*lambda_M_up
            H_k(2,2)               = H_k(2,2)               +   0.0d0
            H_k(3,3)               = H_k(3,3)               + ( 1.0d0)*lambda_M_up
            H_k(4,4)               = H_k(4,4)               + (-1.0d0/2.0d0)*lambda_X2_up
            H_k(5,5)               = H_k(5,5)               +   0.0d0
            H_k(6,6)               = H_k(6,6)               + ( 1.0d0/2.0d0)*lambda_X2_up
            !odd_up
            H_k(7,7)               = H_k(7,7)               + (-1.0d0/2.0d0)*lambda_M_up
            H_k(8,8)               = H_k(8,8)               + ( 1.0d0/2.0d0)*lambda_M_up
            H_k(9,9)               = H_k(9,9)               + (-1.0d0/2.0d0)*lambda_X2_up
            H_k(10,10)             = H_k(10,10)             +  0.0d0
            H_k(11,11)             = H_k(11,11)             + (1.0d0/2.0d0)*lambda_X2_up            
            !even_down
            H_k(1+11,1+11)         = H_k(1+11,1+11)         + (-1.0d0)*lambda_M_down
            H_k(2+11,2+11)         = H_k(2+11,2+11)         +   0.0d0
            H_k(3+11,3+11)         = H_k(3+11,3+11)         + ( 1.0d0)*lambda_M_down
            H_k(4+11,4+11)         = H_k(4+11,4+11)         + (-1.0d0/2.0d0)*lambda_X2_down
            H_k(5+11,5+11)         = H_k(5+11,5+11)         +   0.0d0
            H_k(6+11,6+11)         = H_k(6+11,6+11)         + ( 1.0d0/2.0d0)*lambda_X2_down
            !odd_down
            H_k(7+11,7+11)         = H_k(7+11,7+11)         + (-1.0d0/2.0d0)*lambda_M_down
            H_k(8+11,8+11)         = H_k(8+11,8+11)         + ( 1.0d0/2.0d0)*lambda_M_down
            H_k(9+11,9+11)         = H_k(9+11,9+11)         + (-1.0d0/2.0d0)*lambda_X2_down
            H_k(10+11,10+11)       = H_k(10+11,10+11)       +   0.0d0
            H_k(11+11,11+11)       = H_k(11+11,11+11)       + ( 1.0d0/2.0d0)*lambda_X2_down
            ! spin2
            !even_up            
            H_k(1+22,1+22)         = H_k(1+22,1+22)         - (-1.0d0)*lambda_M_up
            H_k(2+22,2+22)         = H_k(2+22,2+22)         -   0.0d0
            H_k(3+22,3+22)         = H_k(3+22,3+22)         - ( 1.0d0)*lambda_M_up
            H_k(4+22,4+22)         = H_k(4+22,4+22)         - (-1.0d0/2.0d0)*lambda_X2_up
            H_k(5+22,5+22)         = H_k(5+22,5+22)         -   0.0d0
            H_k(6+22,6+22)         = H_k(6+22,6+22)         - ( 1.0d0/2.0d0)*lambda_X2_up
            !odd_up
            H_k(7+22,7+22)         = H_k(7+22,7+22)         - (-1.0d0/2.0d0)*lambda_M_up
            H_k(8+22,8+22)         = H_k(8+22,8+22)         - ( 1.0d0/2.0d0)*lambda_M_up
            H_k(9+22,9+22)         = H_k(9+22,9+22)         - (-1.0d0/2.0d0)*lambda_X2_up
            H_k(10+22,10+22)       = H_k(10+22,10+22)       -  0.0d0
            H_k(11+22,11+22)       = H_k(11+22,11+22)       - (1.0d0/2.0d0)*lambda_X2_up            
            !even_down
            H_k(1+11+22,1+11+22)   = H_k(1+11+22,1+11+22)   - (-1.0d0)*lambda_M_down
            H_k(2+11+22,2+11+22)   = H_k(2+11+22,2+11+22)   -   0.0d0
            H_k(3+11+22,3+11+22)   = H_k(3+11+22,3+11+22)   - ( 1.0d0)*lambda_M_down
            H_k(4+11+22,4+11+22)   = H_k(4+11+22,4+11+22)   - (-1.0d0/2.0d0)*lambda_X2_down
            H_k(5+11+22,5+11+22)   = H_k(5+11+22,5+11+22)   -   0.0d0
            H_k(6+11+22,6+11+22)   = H_k(6+11+22,6+11+22)   - ( 1.0d0/2.0d0)*lambda_X2_down
            !odd_down
            H_k(7+11+22,7+11+22)   = H_k(7+11+22,7+11+22)   - (-1.0d0/2.0d0)*lambda_M_down
            H_k(8+11+22,8+11+22)   = H_k(8+11+22,8+11+22)   - ( 1.0d0/2.0d0)*lambda_M_down
            H_k(9+11+22,9+11+22)   = H_k(9+11+22,9+11+22)   - (-1.0d0/2.0d0)*lambda_X2_down
            H_k(10+11+22,10+11+22) = H_k(10+11+22,10+11+22) -  0.0d0
            H_k(11+11+22,11+11+22) = H_k(11+11+22,11+11+22) - (1.0d0/2.0d0)*lambda_X2_down  
            !even-odd spin mixing
            !layer up
            H_k(1,7+22)  = sqrt(3.0d0/2.0d0)*lambda_M_up
            H_k(2,8+22)  = lambda_M_up
            H_k(4,10+22) = lambda_X2_up/sqrt(2.0d0)
            H_k(5,11+22) = lambda_X2_up/sqrt(2.0d0)
            H_k(7,2+22)  = lambda_M_up
            H_k(8,3+22)  = sqrt(3.0d0/2.0d0)*lambda_M_up
            H_k(9,5+22)  = lambda_X2_up/sqrt(2.0d0)
            H_k(10,6+22) = lambda_X2_up/sqrt(2.0d0)
            !layer down; AB stacking
            H_k(2+11,7+22+11)  = sqrt(3.0d0/2.0d0)*lambda_M_down
            H_k(3+11,8+22+11)  = lambda_M_down
            H_k(4+11,10+22+11) = lambda_X2_down/sqrt(2.0d0)
            H_k(5+11,11+22+11) = lambda_X2_down/sqrt(2.0d0)
            H_k(7+11,1+22+11)  = lambda_M_down
            H_k(8+11,2+22+11)  = sqrt(3.0d0/2.0d0)*lambda_M_down
            H_k(9+11,5+22+11)  = lambda_X2_down/sqrt(2.0d0)
            H_k(10+11,6+22+11) = lambda_X2_down/sqrt(2.0d0)


    end subroutine H_k_MoSe2WSe2

    
!________________________________________________________________________________________________________________________________________________
!________________________________________________________________________________________________________________________________________________
!________________________________________________________________________________________________________________________________________________


program H_k_heterostructures

    implicit none
    
    double precision, parameter  :: pi = 4.0d0*atan(1.0d0)
    
    double precision :: kx, ky, &
                        Ed_up, Ep1_up, Ep0_up, Vdp_sigma_up, Vdp_pi_up, Vdd_sigma_up, Vdd_pi_up, Vdd_delta_up, Vpp_sigma_up, &
                        Vpp_pi_up, Ep1_odd_up, Ep0_odd_up, Ed_odd_up, lambda_M_up, lambda_X2_up, &
                        Ed_down, Ep1_down, Ep0_down, Vdp_sigma_down, Vdp_pi_down, Vdd_sigma_down, Vdd_pi_down, Vdd_delta_down, &
                        Vpp_sigma_down, Vpp_pi_down, Ep1_odd_down, Ep0_odd_down, Ed_odd_down, lambda_M_down, lambda_X2_down, &
                        Vpp_sigma_inter, Vpp_pi_inter, Vdd_sigma_inter, Vdd_pi_inter, Vdd_delta_inter
    double complex  ::  H_k(44,44)
    integer :: ii, jj
    
    

    write(*,*) 'Code started'
    
        ky = 0.0d0
        kx = 4.0d0*pi/(3.0d0*sqrt(3.0d0)*(3.323d0/sqrt(3.0d0)))
    
        ! --- Slater-Koster layers parameters
        ! layer up: MoSe2
        Ed_up          = -0.09d0
        Ep1_up         = -5.01d0
        Ep0_up         = -5.30d0
        Vdp_sigma_up   = -3.08d0
        Vdp_pi_up      =  1.08d0
        Vdd_sigma_up   = -0.94d0
        Vdd_pi_up      =  0.75d0
        Vdd_delta_up   =  0.13d0
        Vpp_sigma_up   =  1.39d0
        Vpp_pi_up      = -0.45d0
        Ep1_odd_up     =  Ep1_up
        Ep0_odd_up     =  Ep0_up
        Ed_odd_up      =  Ed_up
        lambda_M_up    =  0.15d0 ! it should be: 0.186d0/2.0d0
        lambda_X2_up   =  0.15d0 ! it should be: 0.200d0/2.0d0
        ! layer down: WSe2
        Ed_down        = -0.12d0
        Ep1_down       = -4.17d0
        Ep0_down       = -4.52d0
        Vdp_sigma_down = -3.31d0
        Vdp_pi_down    =  1.16d0
        Vdd_sigma_down = -1.14d0
        Vdd_pi_down    =  0.72d0
        Vdd_delta_down =  0.17d0
        Vpp_sigma_down =  1.16d0
        Vpp_pi_down    = -0.25d0
        Ep1_odd_down   =  Ep1_down
        Ep0_odd_down   =  Ep0_down
        Ed_odd_down    =  Ed_down 
        lambda_M_down  =  0.35d0 ! it should be: 0.472d0/2.0d0
        lambda_X2_down = -0.20d0 ! it should be: -0.390d0/2.0d0
        ! interlayer
        Vpp_sigma_inter =  0.5d0 ! positive
        Vpp_pi_inter    = -0.5d0 ! negative
        Vdd_sigma_inter = -0.3d0 ! negative
        Vdd_pi_inter    =  0.3d0 ! positive
        Vdd_delta_inter = -0.6d0 ! negative
      
    call H_k_MoSe2WSe2(kx,ky,Ed_up,Ep1_up,Ep0_up,Vdp_sigma_up,Vdp_pi_up,&
    Vdd_sigma_up,Vdd_pi_up,Vdd_delta_up,Vpp_sigma_up,Vpp_pi_up,Ep1_odd_up,&
    Ep0_odd_up,Ed_odd_up,lambda_M_up,lambda_X2_up,Ed_down,Ep1_down,Ep0_down,&
    Vdp_sigma_down,Vdp_pi_down,Vdd_sigma_down,Vdd_pi_down,Vdd_delta_down,&
    Vpp_sigma_down,Vpp_pi_down,Ep1_odd_down,Ep0_odd_down,Ed_odd_down,&
    lambda_M_down,lambda_X2_down,Vpp_sigma_inter,Vpp_pi_inter,Vdd_sigma_inter,&
    Vdd_pi_inter,Vdd_delta_inter,H_k)           
	
    write(*,*) 'Code finished'
    do ii = 1,44
        do jj = 1,44
            write(*,*) ii, jj, H_k(ii,jj)
        enddo
    enddo

end program H_k_heterostructures
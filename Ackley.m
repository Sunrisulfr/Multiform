function obj = Ackley(var,M,opt,D_eff)
%Ackley function
%   - var: design variable vector
%   - M: rotation matrix
%   - opt: shift vector
 
    var = 30*var;
    opt = 30*opt;

    var = (M*(var-opt)')';  
    var = var(1:D_eff);
     
    dim = length(var);
 
    sum1 = 0; sum2 = 0;
    for i = 1: dim
        sum1 = sum1 + var(i)*var(i);
        sum2 = sum2 + cos(2*pi*var(i));
    end
    avgsum1 = sum1/dim;
    avgsum2 = sum2/dim;
     
    obj = -20*exp(-0.2*sqrt(avgsum1)) - exp(avgsum2) + 20 + exp(1);    
end
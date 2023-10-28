function obj = Weierstrass(var,M,opt,D_eff)
%WEIERSTASS function
%   - var: design variable vector
%   - M: rotation matrix
%   - opt: shift vector

    var = 0.5*var;
    opt = 0.5*opt;
    var = (M*(var-opt)')';
    var = var(1:D_eff);
    
    a = 0.5;
    b = 3;
    kmax = 20;
    obj = 0;
    dim = length(var);

    for i = 1:dim
        for k = 0:kmax
            obj = obj + a^k*cos(2*pi*b^k*(var(i)+0.5));
        end
    end
    for k = 0:kmax
        obj = obj - dim*a^k*cos(2*pi*b^k*0.5);
    end
end
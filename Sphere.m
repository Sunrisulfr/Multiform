function obj = Sphere(var,M,opt,D_eff)
%Sphere function
%   - var: design variable vector
%   - opt: shift vector

    var = 5*var;
    opt = 5*opt;
    var = (M*(var-opt)')';  
    var = var(1:D_eff);
    dim = length(var);
    obj = 0;
    a = 1e+6;
    for i=1:dim
        obj = obj + a.^((i-1)/(dim-1)).*var(i).^2; 
    end
end
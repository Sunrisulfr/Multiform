function obj = Rastrigin(var,M,opt,D_eff)
%Rastrigin function
%   - var: design variable vector
%   - M: rotation matrix
%   - opt: shift vector
    var = 5*var;
    opt = 5*opt;
    var = (M*(var-opt)')';
    var = var(1:D_eff);

    dim = length(var);

    obj = 10*dim;
    for i=1:dim
        obj=obj+(var(i)^2 - 10*(cos(2*pi*var(i))));
    end
end
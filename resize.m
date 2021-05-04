function A = resize(out)

[m, n, k] = size(out);
A_temp = zeros(m+2,n+2);

A_temp(2:m+1, 2:n+1) = out;

A = A_temp;
end
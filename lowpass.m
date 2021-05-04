function [sl, sh] = lowpass(s, lambda, npad, ratio)

if nargin < 3
  npad = 16;
end

grv = [-1 1];
gcv = [-1 1]';

Gr = fft2(grv, size(s,1)+2*npad, size(s,2)+2*npad);
Gc = fft2(gcv, size(s,1)+2*npad, size(s,2)+2*npad);

A = ratio + lambda*conj(Gr).*Gr + lambda*conj(Gc).*Gc;

sp = padarray(s, [npad npad], 'symmetric', 'both');
slp = ifft2(bsxfun(@rdivide, fft2(sp), A), 'symmetric');

sl = slp((npad+1):(size(slp,1)-npad), (npad+1):(size(slp,2)-npad), :);
sh = s - sl;

return

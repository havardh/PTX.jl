using BenchmarkLite

import OpenCLBlur
import JuliaBlur
import CUDABlur

procs = Proc[
  JuliaBlur.Benchmark(),
  CUDABlur.Benchmark(),
  OpenCLBlur.Benchmark()
];

cfgs = 2 .^ [11:14];

rtable = run(procs, cfgs);

show(rtable; unit=:msec);


function save_plot(b::BenchmarkTable, filename)

  sizes = b.cfgs

  plot()
  for i in 1:size(b.etime, 2)


    oplot(sizes, (b.etime[:,i] ./ b.nruns[:,i]));
  end

  savefig(filename)

end

save_plot(rtable, "result.png")

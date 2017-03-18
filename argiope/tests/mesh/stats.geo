lc = 1e-2;
Point(1) = {0, 0, 0, lc};
Point(2) = {.1, 0,  0, lc} ;
Point(3) = {.1, .3, 0, lc} ;
Point(4) = {0,  .3, 0, lc} ;
Line(1) = {1,2} ;
Line(2) = {3,2} ;
Line(3) = {3,4} ;
Line(4) = {4,1} ;
Line Loop(5) = {4,1,-2,3} ;
Plane Surface(6) = {5} ;

// mesh
Mesh 2;

// what to print:
Print.PostElementary = 1;
Print.PostElement = 0;
Print.PostGamma = 0;
Print.PostEta = 0;
Print.PostRho = 1;
Print.PostDisto = 0;

// save (using automatic format detection using the extension)
Save "aa.pos";

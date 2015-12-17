package glaux

import glaux.linearalgebra.impl.nd4j.ND4JImplementation

//The single link towards the implementation linalg data structure.
package object linearalgebra extends ND4JImplementation with Implementation {

}

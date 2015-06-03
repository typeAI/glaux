package glaux

import glaux.linalg.impl.nd4j.ND4JImplementation

//The single link towards the implementation linalg data structure.
package object linalg extends ND4JImplementation with Implementation {

}

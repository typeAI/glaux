package glaux.neuralnetwork.layers

import glaux.linearalgebra.Dimension.ThreeD
import glaux.linearalgebra.Vol
import glaux.neuralnetwork.{Gradient, Rectangle}
import org.specs2.mutable.Specification

class PoolSpec extends Specification {
  "creation" >> {
    "padding" >> {
      Pool(
        ThreeD(4, 4, 1),
        Rectangle(3, 3),
        3,
        true
      ).pad === Rectangle(1, 1)

      Pool(
        ThreeD(4, 4, 1),
        Rectangle(2, 2),
        2,
        true
      ).pad === Rectangle(0, 0)

    }

  }

  val expectedOutput = Vol(2, 2, 1, Seq(6, 8, 9, 7))
  val input = Vol(4, 4, 1, Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7))

  "forward" >> {
    val l = Pool(ThreeD(4, 4, 1), Rectangle(2, 2), 2)
    l.forward(input, false) === expectedOutput
  }

  "backward" >> {
    val expected = Vol(4, 4, 1, Seq(0, 0, 0, 0, 0, 1, 0, 4, -1, 0, 0, 0, 0, 0, 0, 2))
    val l = Pool(ThreeD(4, 4, 1), Rectangle(2, 2), 2)

    val (inGrad, _) = l.backward(input, Gradient(expectedOutput, Vol(2, 2, 1, Seq(1, 4, -1, 2))))
    inGrad === Vol(4, 4, 1, Seq(0, 0, 0, 0, 0, 1, 0, 4, -1, 0, 0, 0, 0, 0, 0, 2))
  }

}

/* convnet js test code

  function linear(vol){
    var rv = []
    for(z = 0; z < vol.depth ; z ++ ) {
      for(y = 0; y < vol.sy ; y ++ ) {
        for(x = 0; x < vol.sx ; x ++ ) {
          rv.push(vol.get(x, y, z));
        }
      }
    }
    return rv;
  }

  function linearGrad(vol){
    var rv = []
    for(z = 0; z < vol.depth ; z ++ ) {
      for(y = 0; y < vol.sy ; y ++ ) {
        for(x = 0; x < vol.sx ; x ++ ) {
          rv.push(vol.get_grad(x, y, z));
        }
      }
    }
    return rv;
  }

  var l = new convnetjs.PoolLayer({in_sx: 4, in_sy: 4, in_depth: 1, stride: 2, sx: 2, sy: 2, pad: 0});
  var input = new convnetjs.Vol(4, 4, 1, 1)
  input.w = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7]
  linear(input)
  var output = l.forward(input)

  output.w


  output.dw = [1,4,-1,2]
  l.backward()
  l.in_act.dw
 */

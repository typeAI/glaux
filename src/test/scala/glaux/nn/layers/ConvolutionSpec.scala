package glaux.nn.layers

import glaux.linalg._
import Dimension._
import glaux.nn.Rectangle
import org.specs2.mutable.Specification
import Convolution._
class ConvolutionSpec extends Specification {
  "creation" >> {
    "without padding" >> {
      "reduced out dimension" >> {
        val inputDim = ThreeD(4, 4, 2)
        val layer = Convolution(3, Rectangle(3, 3), inputDim, 1, false)
        layer.outDimension === ThreeD(2, 2, 3)
      }
    }
    "with padding" >> {
      "same output dimension as input" >> {
        val inputDim = ThreeD(4, 4, 2)
        val layer = Convolution(3, Rectangle(3, 3), inputDim, 1, true)
        layer.outDimension === ThreeD(4, 4, 3)
      }
    }

  }


  "forward" >> {
    "calculate output correctly" >> {
      val inputDim = ThreeD(4, 4, 2)
      val layer = Convolution(3, Rectangle(3, 3), inputDim, 1, true).copy (
        filters = Tensor4( 3, 3, 2, 3, Seq(0.3, 0.1, -0.1, -0.1, 0.3, 0.1, 0.3, 0.4, -0.1, -0.4, 0.2, -0.03, 0.2, 0.4, 0.3, 0.1, 0.2, 0.1, 0, 0.2, -0.2, -0.1, -0.2, -0.1, -0.1, 0.2, 0.3, -0.2, 0.2, 0.1, 0.3, -0.3, 0.3, -0.1, 0.1, -0.3, -0.4, -0.2, -0.2, -0.1, -0.1, 0.1, 0.3, 0.3, -0.1, -0.5, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1))
      )
      val input = Vol(inputDim, Seq(1, 2, 5, 3, 9, 1, 5, 1, 4, 11, 7, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 9, 1))
      val output = layer.forward(input, false)
      output === Vol(4, 4, 3, Seq(5, 4.9, 5.1, 3.2, 4.37, 5.87, 9.17, 5.6, 7.27, 11.27, 5.97, 4, 4.17, 7.27, 9.27, 4.6, 1.5, -0.2, -0.3, -1.4, 2.1, 1.8, 0.5, -0.2, -0.8, -5.8, -1.4, -1.8, -0.3, 0.6, 0.5, 2.7, 2.7, 2.9, 1.5, 1, -1.3, 1.2, 1.9, -0.8, 1.1, -3.8, -3.0, -2.4, -2.8, -6.1, -7.0, -3.4))
    }
  }

}

/* convnetjs test code
  var l = new convnetjs.ConvLayer({in_sx: 4, in_sy: 4, in_depth: 2, filters: 3, sx: 3, sy: 3, stride: 1, pad: 1});
  var input = new convnetjs.Vol(4, 4, 2, 1)
  input.w = [1, 1, 2, 1, 5, 1, 3, 1, 9, 1, 1, 1, 5, 1, 1, 1, 4, 1, 11, 1, 7, 1, 3, 1, 6, 1, 1, 8, 1, 9, 1, 1]
  var ir = [] //input in linear array
  for(z = 0; z < 2 ; z ++ ) {
    for(y = 0; y < 4 ; y ++ ) {
      for(x = 0; x < 4 ; x ++ ) {
        ir.push(input.get(x, y, z))
      }
    }
  }

  l.filters[0].w = [0.3, -0.4, 0.1, 0.2, -0.1, -0.03, -0.1, 0.2, 0.3, 0.4, 0.1, 0.3, 0.3, 0.1, 0.4, 0.2, -0.1, 0.1]
  l.filters[1].w = [0, -0.2, 0.2, 0.2, -0.2, 0.1, -0.1, 0.3, -0.2, -0.3, -0.1, 0.3, -0.1, -0.1, 0.2, 0.1, 0.3, -0.3]
  l.filters[2].w = [-0.4, -0.5, -0.2, -0.1, -0.2, 0.1, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.3, 0.1, 0.3, -0.1, -0.1, 0.1]

  var f = [] //filter in linear array
  for(d = 0; d < 3 ; d ++ ) {
    for(z = 0; z < 2 ; z ++ ) {
      for(y = 0; y < 3 ; y ++ ) {
        for(x = 0; x < 3 ; x ++ ) {
          f.push(l.filters[d].get(x, y, z))
        }
      }
    }
  }
  f

  var output = l.forward(input);

  var or = [] //output in linear
  for(z = 0; z < 3 ; z ++ ) {
    for(y = 0; y < 4 ; y ++ ) {
      for(x = 0; x < 4 ; x ++ ) {
        or.push(output.get(x, y, z))
      }
    }
  }

*/

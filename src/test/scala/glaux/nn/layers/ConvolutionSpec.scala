package glaux.nn.layers

import glaux.linalg._
import Dimension._
import glaux.nn.{Gradient, Rectangle}
import org.specs2.mutable.Specification
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

  "backward/forward" >> {
    val inputDim = ThreeD(4, 4, 2)
    val layer = Convolution(3, Rectangle(3, 3), inputDim, 1, true).copy (
      filters = Tensor4( 3, 3, 2, 3, Seq(0.3, 0.1, -0.1, -0.1, 0.3, 0.1, 0.3, 0.4, -0.1, -0.4, 0.2, -0.03, 0.2, 0.4, 0.3, 0.1, 0.2, 0.1, 0, 0.2, -0.2, -0.1, -0.2, -0.1, -0.1, 0.2, 0.3, -0.2, 0.2, 0.1, 0.3, -0.3, 0.3, -0.1, 0.1, -0.3, -0.4, -0.2, -0.2, -0.1, -0.1, 0.1, 0.3, 0.3, -0.1, -0.5, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1))
    )
    val input = Vol(inputDim, Seq(1, 2, 5, 3, 9, 1, 5, 1, 4, 11, 7, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 9, 1))

    "forward" >> {
      val output = layer.forward(input, false)
      output === Vol(4, 4, 3, Seq(5, 4.9, 5.1, 3.2, 4.37, 5.87, 9.17, 5.6, 7.27, 11.27, 5.97, 4, 4.17, 7.27, 9.27, 4.6, 1.5, -0.2, -0.3, -1.4, 2.1, 1.8, 0.5, -0.2, -0.8, -5.8, -1.4, -1.8, -0.3, 0.6, 0.5, 2.7, 2.7, 2.9, 1.5, 1, -1.3, 1.2, 1.9, -0.8, 1.1, -3.8, -3.0, -2.4, -2.8, -6.1, -7.0, -3.4))
    }

    "backward" >> {
      val output = layer.forward(input, false)

      val outGrad = Vol(4, 4, 3, Seq(0, 0, 0, 0, 0.9, 0, 3, 0, 0, 0, 0, -1, 0, 0, 0.2, 0.3, 0.4, 0.1, 0, 0, 0, 2, 0.3, 0.09, 0, 0, 0, -0.3, 0.4, 0.7, 0.1, 0.5, 0.2, -0.1, -0.01, 0.3, 1, 0, 0, -1, 0, 0, 3, 0, 0, 0, 0, 0))
      val (inGrad, paraGrads) = layer.backward(input, Gradient(output, outGrad ))

      val expectedInGrad = Vol(inputDim, Seq(-0.21, 0.99, 0.311, -0.173, 0.07, -1.653, -0.042, -0.317, 0.54, 0.9, 1.371, -0.342, -0.15, 0.69, 0.52, -0.76, -0.44, -0.668, 1.393, 0.027, 0.86, -1.121, 1.788, 1.072, -0.18, 1.04, -0.759, 0.473, 0.09, 0.28, 0.1, 0.13))
      expectedInGrad === inGrad
    }
  }



}

/* convnetjs test code


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


  var l = new convnetjs.ConvLayer({in_sx: 4, in_sy: 4, in_depth: 2, filters: 3, sx: 3, sy: 3, stride: 1, pad: 1});
  var input = new convnetjs.Vol(4, 4, 2, 1)
  input.w = [1, 1, 2, 1, 5, 1, 3, 1, 9, 1, 1, 1, 5, 1, 1, 1, 4, 1, 11, 1, 7, 1, 3, 1, 6, 1, 1, 8, 1, 9, 1, 1]
  linear(input)

  l.filters[0].w = [0.3, -0.4, 0.1, 0.2, -0.1, -0.03, -0.1, 0.2, 0.3, 0.4, 0.1, 0.3, 0.3, 0.1, 0.4, 0.2, -0.1, 0.1]
  l.filters[1].w = [0, -0.2, 0.2, 0.2, -0.2, 0.1, -0.1, 0.3, -0.2, -0.3, -0.1, 0.3, -0.1, -0.1, 0.2, 0.1, 0.3, -0.3]
  l.filters[2].w = [-0.4, -0.5, -0.2, -0.1, -0.2, 0.1, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.3, 0.1, 0.3, -0.1, -0.1, 0.1]

  var f = [] //filter in linear array
  for(d = 0; d < 3 ; d ++ ) {
    f = f.concat(linear(l.filters[d])
  }
  f

  var output = l.forward(input);

  linear(output)


  l.out_act.dw = [0, 0.4, 0.2, 0, 0.1, -0.1, 0, 0, -0.01, 0, 0, 0.3, 0.9, 0, 1, 0, 2, 0, 3, 0.3, 0, 0, 0.09, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -1, -0.3, 0, 0, 0.4, 0, 0, 0.7, 0, 0.2, 0.1, 0, 0.3, 0.5, 0]

  linearGrad(out_act)

  l.backward()

  linearGrad(l.in_act)


*/

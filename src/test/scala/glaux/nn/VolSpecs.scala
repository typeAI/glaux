package glaux.nn

import glaux.nn.Dimension.Dimension3D
import org.nd4j.linalg.factory.Nd4j
import org.specs2.mutable.Specification

class VolSpecs extends Specification {

  "3D Vols" >> {
    val dim = Dimension3D(1,1,3)

    "* matrix keeps dimension" >> {
      val vol = Vol(dim, 2, 3, 4)
      val vol2: Vol = vol * vol
      vol2 must_== Vol(dim, 4, 9, 16)
    }

    "* scalar keeps dimension" >> {
      val vol = Vol(dim, 2, 3, 4)
      val vol2: Vol = vol * 0.5
      vol2 must_== Vol(dim, 1, 1.5, 2)
    }

  }

  "1D Vols" >> {
    "sum" >> {
      Vol1D(1, 3, 4, 5).sum === 13
    }
    "*" >> {
      val result: Vol = Vol1D(1, 3, 4, 5) * 3
      result must_==(Vol1D(3, 9, 12, 15))
    }


  }
}

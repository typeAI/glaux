package glaux.nn

import glaux.nn.Dimension.{ThreeD}
import org.specs2.mutable.Specification

class VolSpecs extends Specification {

  "3D Vols" >> {
    val dim = ThreeD(1,1,3)

    "* op keeps dimension" >> {
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
    "sum works correctly" >> {
      RowVector(1, 3, 4, 5).sum === 13
    }
    "* works correctly" >> {
      val result: Vol = RowVector(1, 3, 4, 5) * 3
      result must_==(RowVector(3, 9, 12, 15))
    }


  }
}

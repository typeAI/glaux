package glaux.nn

import glaux.nn.Dimension.{Row, TwoD, ThreeD}
import org.specs2.mutable.Specification

class VolSpec
  extends Specification {



  "3D Vols" >> {
    val dim = ThreeD(1,1,3)

    "* op keeps dimension" >> {
      val vol = Vol(dim, Seq(2, 3, 4))
      val vol2: Vol = vol * vol
      vol2 must_== Vol(dim, Seq(4, 9, 16))
    }

    "* scalar keeps dimension" >> {
      val vol = Vol(dim, Seq(2, 3, 4))
      val vol2: Vol = vol * 0.5
      vol2 must_== Vol(dim, Seq(1, 1.5, 2))
    }

    "sum correctly" >> {
      val m = Matrix(2, 2, Seq(1,2,3,4))
      m.sumAll must_== 10
    }

    "map" >> {
      "with immutability" >> {
        val m = Matrix(2, 2, Seq(1,2,3,4))
        m.map(_ * 2)
        m must_== Matrix(2, 2, Seq(1,2,3,4))
      }

      "correctly" >> {
        val m = Matrix(2, 2, Seq(1,2,3,4))
        val result = m.map( _ * 2 )
        result must_== Matrix(2, 2, Seq(2,4,6,8))
      }
    }

    "merge" >> {

      "correctly" >> {
        val m = Vol3D(ThreeD(2, 2, 2), 1.until(9).map(_.toDouble))
        val m2 = Vol3D(ThreeD(2, 2, 2), 9.until(1, -1).map(_.toDouble))
        val result = m.merge(m2)(_ + _)
        result must_== Vol3D(ThreeD(2, 2, 2), Seq.fill(8)(10d))
      }
    }

    "uniform" >> {
      val m = Vol3D.uniform(ThreeD(2, 2, 2), 10)
      m must_== Vol3D(2, 2, 2, Seq.fill(8)(10d))

    }

    "normal" >> {
      val data = 1.until(100).flatMap { _ =>
        RowVector.normal(Row(3), 10, 3).toArray.toSeq
      }
      val avg = data.sum / data.size
      val devs = data.map(value => (value - avg) * (value - avg))
      val std =  Math.sqrt(devs.sum / data.size)
      avg must beCloseTo(10.0 within 2.significantFigures)
      std must beCloseTo(3.0 within 1.significantFigures)
    }

  }

  "Generic " >> {
    "map correctly" >> {
      val d: TwoD = TwoD(3, 1)
      val m: Vol = Matrix.uniform(d, 0.5)
      val result = m.map(_ * 2)
      result must_== Matrix.uniform(d, 1)
    }
  }

  "RowVector Vols" >> {
    "sum works correctly" >> {
      RowVector(1, 3, 4, 5).sumAll === 13
    }
    "* works correctly" >> {
      val result: Vol = RowVector(1, 3, 4, 5) * 3
      result must_==(RowVector(3, 9, 12, 15))
    }


  }


}

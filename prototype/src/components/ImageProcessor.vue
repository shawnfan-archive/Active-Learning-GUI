<template>
  <v-card class="ma-1 pa-0">
    <v-container wrap class="px-2 py-0">
      <v-row no-gutters align="center" class="ma-2">
        <v-col cols="3">
          <div class="histogram">
            <canvas ref="histogram_curr" width="150" height="150"></canvas>
          </div>
        </v-col>
        <v-col cols="5">
          <v-row no-gutters class="ma-2">
            <v-col cols="12">
              <v-slider v-model="brightness" min="0" max="200" label="Brightness">
                <template v-slot:append>
                  <v-text-field
                    v-model="brightness"
                    class="mt-0 pt-0"
                    hide-details
                    single-line
                    type="number"
                    style="width: 50px"
                  ></v-text-field>
                </template>
              </v-slider>
            </v-col>
            <v-col cols="12">
              <v-slider v-model="gamma" min="0" max="300" label="Gamma">
                <template v-slot:append>
                  <v-text-field
                    v-model="gamma"
                    class="mt-0 pt-0"
                    hide-details
                    single-line
                    type="number"
                    style="width: 50px"
                  ></v-text-field>
                </template>
              </v-slider>
            </v-col>
            <v-col cols="12">
              <v-range-slider
                v-model="pixel_range"
                max="10000"
                min="0"
                hide-details
                class="align-center"
                label="Pixel Range"
              >
                <template v-slot:prepend>
                  <v-text-field
                    v-model="pixel_range[0]"
                    class="mt-0 pt-0"
                    hide-details
                    single-line
                    type="number"
                    style="width: 60px"
                  ></v-text-field>
                </template>
                <template v-slot:append>
                  <v-text-field
                    v-model="pixel_range[1]"
                    class="mt-0 pt-0"
                    hide-details
                    single-line
                    type="number"
                    style="width: 60px"
                  ></v-text-field>
                </template>
              </v-range-slider>
            </v-col>
            <v-col cols="6">
              <v-btn
                class="ma-0 pa-0 success"
                dark
                outlined
                small
                text
                v-on:click="saveChanges"
              >Save</v-btn>
            </v-col>
            <v-col cols="6">
              <v-btn
                class="ma-0 pa-0 success"
                dark
                outlined
                small
                text
                v-on:click="resetImage"
              >Reset</v-btn>
            </v-col>
          </v-row>
        </v-col>
        <v-col cols="4">
          <!-- <v-sheet class="ma-4" elevation color="transparent"> -->
          <div class="histogram">
            <canvas ref="histogram_main" width="500" height="150"></canvas>
          </div>
          <!-- </v-sheet> -->
        </v-col>
      </v-row>
    </v-container>
  </v-card>
</template>

<script>
import Chart from "chart.js";

export default {
  props: {
    image: Array,
    h: Number,
    w: Number
  },
  data() {
    return {
      brightness: 100,
      gamma: 100,
      total_range: [0, 0],
      pixel_range: [0, 100],
      current_chart: Object,
      current_data: new Array(256).fill(0),
      main_chart: Object,
      main_data: Array,
      adjusted_img: new Array(this.h).fill(0).map(x => Array(this.w).fill(0))
    };
  },
  methods: {
    initializeParams: function() {
      // Find max range of pixel values
      let minRow = this.image.map(function(row) {
        return Math.min.apply(Math, row);
      });
      let maxRow = this.image.map(function(row) {
        return Math.max.apply(Math, row);
      });
      this.total_range[0] = Math.min.apply(null, minRow);
      this.total_range[1] = Math.max.apply(null, maxRow);
      this.pixel_range = this.total_range;

      // Data for main histogram
      this.main_data = new Array(this.total_range[1]).fill(0);
      for (let i = 0; i < this.h; i += 1) {
        for (let j = 0; j < this.w; j += 1) {
          this.main_data[Math.floor(this.image[i][j])] += 1;
        }
      }

      // Adjusted image
      for (let i = 0; i < this.h; i += 1) {
        for (let j = 0; j < this.w; j += 1) {
          this.adjusted_img[i][j] = Math.floor(
            this.stretchContrast(this.image[i][j])
          );
        }
      }
      console.log(this.adjusted_img);
    },
    resetImage: function() {
      this.pixel_range = this.total_range;
      this.gamma = 100;
      this.brightness = 100;
      //   this.adjustImage();
    },
    createHistograms: function() {
      // Create hitograms at creation of component
      let main_ctx = this.$refs.histogram_main.getContext("2d");

      let current_ctx = this.$refs.histogram_curr.getContext("2d");

      this.main_chart = new Chart(main_ctx, {
        type: "line",
        data: {
          datasets: [
            {
              data: [],
              fill: true,
              borderColor: "rgb(63, 191, 289)",
              backgroundColor: "rgba(63, 191, 289, 0.1)",
              borderWidth: 0,
              pointRadius: 0
            },
            {
              data: [],
              fill: true,
              borderColor: "rgb(63, 191, 289)",
              backgroundColor: "rgba(63, 191, 289, 0.8)",
              borderWidth: 0,
              pointRadius: 0
            },
            {
              data: [],
              fill: true,
              borderColor: "rgb(63, 191, 289)",
              backgroundColor: "rgba(63, 191, 289, 0.1)",
              borderWidth: 0,
              pointRadius: 0
            }
          ]
        },
        options: {
          animation: {
            duration: 0
          },
          hover: {
            animationDuration: 0 // duration of animations when hovering an item
          },
          //Responsive must be true to allow plot to rescale
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            xAxes: [
              {
                type: "linear"
              }
            ],
            yAxes: [
              {
                type: "logarithmic",
                ticks: {
                  //Hide ticks on y-axis
                  display: false
                }
              }
            ]
          },
          legend: {
            //Remove legend
            display: false
          }
        }
      });

      this.current_chart = new Chart(current_ctx, {
        type: "line",
        data: {
          datasets: [
            {
              data: [],
              fill: true,
              borderColor: "rgb(63, 191, 289)",
              backgroundColor: "rgba(63, 191, 289, 0.1)",
              borderWidth: 1,
              pointRadius: 1
            }
          ]
        },
        options: {
          animation: {
            duration: 0
          },
          hover: {
            animationDuration: 0 // duration of animations when hovering an item
          },
          //Responsive must be true to allow plot to rescale
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            //! xAxes type must be None to display data properly
            xAxes: [
              {
                ticks: {
                  display: true,
                  //Use callback to only display the first and last label
                  callback: function(value, index, values) {
                    if (index === 0 || index === values.length - 1) {
                      return String(value);
                    } else {
                      return "";
                    }
                  }
                }
              }
            ],
            yAxes: [
              {
                type: "logarithmic",
                ticks: {
                  //Hide ticks on y-axis
                  display: false
                }
              }
            ]
          },
          legend: {
            //Remove legend
            display: false
          }
        }
      });
    },
    updateMainHistogram: function() {
      let histogram_data_0 = [];
      let histogram_data_1 = [];
      let histogram_data_2 = [];

      for (let i = this.total_range[0]; i < this.pixel_range[0]; i += 1) {
        histogram_data_0.push({ x: i, y: this.main_data[i] });
      }
      this.main_chart.data.datasets[0].data = histogram_data_0;

      for (let i = this.pixel_range[0]; i <= this.pixel_range[1]; i += 1) {
        histogram_data_1.push({ x: i, y: this.main_data[i] });
      }
      this.main_chart.data.datasets[1].data = histogram_data_1;

      for (let i = this.pixel_range[1] + 1; i < this.total_range[1]; i += 1) {
        histogram_data_2.push({ x: i, y: this.main_data[i] });
      }
      this.main_chart.data.datasets[2].data = histogram_data_2;

      this.main_chart.update();
    },
    updateCurrentHistogram: function() {
      console.log(this.adjusted_img);
      this.current_data.fill(0);
      for (let i = 0; i < this.h; i += 1) {
        for (let j = 0; j < this.w; j += 1) {
          this.current_data[Math.floor(this.adjusted_img[i][j])] += 1;
        }
      }
      this.current_chart.data.datasets[0].data = this.current_data;
      this.current_chart.update();
    },
    adjustImage: function() {
      this.pointOperation(this.stretchContrast);
      //! Gamma and brightness must be adjusted after histogram equalization
      if (this.gamma !== 100) {
        this.pointOperation(this.adjustGamma);
      }
      if (this.brightness !== 100) {
        this.pointOperation(this.adjustBrightness);
      }
    },
    pointOperation: function(adjust) {
      // Point operation
      for (let i = 0; i < this.h; i += 1) {
        for (let j = 0; j < this.w; j += 1) {
          this.adjusted_img[i][j] = adjust(this.adjusted_img[i][j]);
        }
      }
    },
    limitRange: function(pixel) {
      //Constrain pixel value to [0,255]
      if (pixel < 0) {
        pixel = 0;
      }
      if (pixel > 255) {
        pixel = 255;
      }
      return pixel;
    },
    adjustBrightness: function(pixel) {
      // Point operation for brightness adjustment
      pixel = (this.brightness / 100) * pixel;
      return this.limitRange(pixel);
    },
    adjustGamma: function(pixel) {
      // Point operation for gamma adjustment
      pixel = 255 * Math.pow(pixel / 255, this.gamma / 100);
      return this.limitRange(pixel);
    },
    stretchContrast: function(pixel) {
      // Point operation for contrast stretching
      pixel =
        ((pixel - this.pixel_range[0]) * (255 - 0)) /
        (this.pixel_range[1] - this.pixel_range[0]);
      console.log(pixel);
      return this.limitRange(pixel);
    },
    saveChanges: function() {
      this.adjustImage();
      this.updateMainHistogram();
      this.updateCurrentHistogram();
      this.$emit("process_image", this.adjusted_img);
    }
  },
  created() {
    this.initializeParams();
  },
  mounted() {
    console.log(this.image);
    this.createHistograms();
    console.log(this.main_data);
    this.adjustImage();
    this.updateMainHistogram();
    this.updateCurrentHistogram();
    console.log(this.current_data);
  }
};
</script>
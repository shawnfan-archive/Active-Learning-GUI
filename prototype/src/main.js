import Vue from 'vue'
import App from './App.vue'
import Router from './router.js'
import VueKonva from 'vue-konva'

Vue.config.productionTip = false

Vue.use(VueKonva)

new Vue({
  router: Router,
  template: "<App/>",
  render: h => h(App)
}).$mount('#app')
